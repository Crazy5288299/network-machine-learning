import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,JumpingKnowledge
from torch_geometric.utils.dropout import dropout_adj
from torch.nn import ModuleList,Linear
import numpy as np 
import random 
from optim.sam import SAM
from torch_geometric.nn.models import CorrectAndSmooth

device = torch.device('cuda:0')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(42)

class Net(torch.nn.Module):
    def __init__(self,hidden_nums = 128,num_layers=8,mode='cat'):
        super(Net, self).__init__()

        self.convs = []
        for i in range(num_layers):
            if i==0:
                mo = GCNConv(1433,hidden_nums)
            else:
                mo = GCNConv(hidden_nums,hidden_nums)
            self.convs.append(mo)
        self.convs = ModuleList(self.convs)
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin = Linear(hidden_nums*num_layers,7)
        else:
            self.lin = Linear(hidden_nums,7)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        xs = []
        for mo in self.convs:
            x = mo(x,edge_index)
            x = F.relu(x)
            x = F.dropout(x,training=self.training,p=0.8)
            edge_index,_ = dropout_adj(edge_index,training=self.training,p=0.3)
            xs += [x] 
        
        x = self.jump(xs)
        x = self.lin(x)
        
        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='/datasets/Cora', name='Cora')

GCN = Net().to(device)
data = dataset[0].to(device)
#optimizer = torch.optim.Adam(GCN.parameters(), lr=1e-3, weight_decay=5e-3)
optimizer = SAM(GCN.parameters(), torch.optim.Adam, rho=0.5, adaptive=True, lr=3e-3, weight_decay=5e-3)

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

print('acc before cs algotithm:', best_acc)
post = CorrectAndSmooth(num_correction_layers=5, correction_alpha=1.0,
                        num_smoothing_layers=5, smoothing_alpha=0.8,
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

#jk: 0.8119999766349792
#jk+drop edge: GCN4: 0.8220000267028809  GCN8: 0.8270000219345093

#400 epoch: GCN4: 0.8299999833106995  GCN8: 0.828000009059906

#发现使用jknet等技术优化后，可以加入很大比例的dropout和dropedge
#可以使用更长的epoch进行训练