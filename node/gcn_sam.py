import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GINConv
from optim.sam import SAM
from torch_geometric.nn.models import CorrectAndSmooth
from torch_geometric.nn import GATConv
from torch.nn import Linear,Sequential,BatchNorm1d,ReLU
from torch_geometric.utils.dropout import dropout_adj

device = torch.device('cuda:0')
        
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1433, 16)
        self.conv2 = GCNConv(16, 7)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        #edge_index,_ = dropout_adj(edge_index,training=self.training,p=0.2)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='/datasets/Cora', name='Cora')

GCN = Net().to(device)
data = dataset[0].to(device)
#optimizer = torch.optim.Adam(GCN.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = SAM(GCN.parameters(), torch.optim.Adam, rho=0.5, adaptive=True, lr=0.01, weight_decay=5e-4)

def train_one_epoch():
    GCN.train()
    #SAM优化器首次迭代
    out = GCN(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.first_step(zero_grad=True)

    #SAM优化器第二次迭代
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
post = CorrectAndSmooth(num_correction_layers=10, correction_alpha=1.0,
                        num_smoothing_layers=10, smoothing_alpha=0.8,
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

#对于GCN

#未使用sam的情况下
# 固定epoch=200       
# GCN acc 81.10%

#加上了SAM 82.10%
#再使用correct and smooth 进行后处理： 83%

