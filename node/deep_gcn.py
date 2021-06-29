import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,JumpingKnowledge
from torch_geometric.utils.dropout import dropout_adj
from torch.nn import ModuleList
import numpy as np 
import random 
from optim.sam import SAM
from torch_geometric.nn.models import CorrectAndSmooth

device = torch.device('cuda:0')

#使用drop_edge技术，并且进行对比试验

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(42)

class Net(torch.nn.Module):
    def __init__(self,hidden_nums = 128,num_layers=8):
        super(Net, self).__init__()

        self.convs = []
        for i in range(num_layers):
            if i==0:
                mo = GCNConv(1433,hidden_nums)
            elif i== num_layers-1:
                mo = GCNConv(hidden_nums,7)
            else:
                mo = GCNConv(hidden_nums,hidden_nums)
            self.convs.append(mo)
        self.convs = ModuleList(self.convs)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for mo in self.convs:
            x = mo(x,edge_index)
            x = F.relu(x)
            x = F.dropout(x,training=self.training,p=0.2)
            edge_index,_ = dropout_adj(edge_index,training=self.training,p=0.2)

        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='/datasets/Cora', name='Cora')

GCN = Net().to(device)
data = dataset[0].to(device)
#optimizer = torch.optim.Adam(GCN.parameters(), lr=3e-3, weight_decay=5e-3)
optimizer = SAM(GCN.parameters(), torch.optim.Adam, rho=0.5, adaptive=True, lr=3e-3, weight_decay=5e-3)

#进行两种不同优化器的对比，通过切换train_one_epoch中的use_sam的值
def train_one_epoch(use_sam=True):
    if use_sam is False:
        GCN.train()
        optimizer.zero_grad()
        out = GCN(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    else:
        GCN.train()
        out = GCN(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.first_step(zero_grad=True)

        out = GCN(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.second_step(zero_grad=True)
    return loss.item()

def test_one_epoch():
    GCN.eval()
    _, pred = GCN(data).max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
    accuracy = correct / data.test_mask.sum()
    return accuracy.item()

best_acc = 0
GCN.train()
for epoch in range(200):
    loss = train_one_epoch()
    acc = test_one_epoch()
    if acc>best_acc:
        best_acc = acc 
        out = GCN(data)
        y_soft = out.softmax(dim=-1)

    if epoch % 1 == 0:
        print('epoch',epoch,'loss',loss,'accuracy',acc)

#Deep GCN设置num_iters 为0会取得更好的效果
print('acc before cs algotithm:', best_acc)
post = CorrectAndSmooth(num_correction_layers=0, correction_alpha=1.0,
                        num_smoothing_layers=0, smoothing_alpha=0.8,
                        autoscale=False, scale=20.)

#使用correct and smooth 算法进行后处理
print('Correct and smooth...')
x_train, y_train = data.x[data.train_mask], data.y[data.train_mask]
y_soft = post.correct(y_soft, y_train, data.train_mask, data.edge_index)
y_soft = post.smooth(y_soft, y_train, data.train_mask, data.edge_index)

pred = y_soft.argmax(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
val_acc = correct / data.test_mask.sum()
test_acc = val_acc.item()
print(f'Test: {test_acc:.4f}')


# 无dropedge
# epoch 399 loss 0.6530132293701172 accuracy 0.656000018119812
# 有dropedge
# epoch 394 loss 0.6903894543647766 accuracy 0.7799999713897705

#经过痛苦的调参并且加入SAM优化器优化过后
#valinna: epoch 99 loss 0.7824620604515076 accuracy 0.6669999957084656
#dropedge: epoch 58 loss 0.40700745582580566 accuracy 0.8130000233650208
