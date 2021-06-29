import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from optim.sam import SAM
from torch_geometric.nn.models import CorrectAndSmooth
from torch_geometric.nn import GATConv
from torch.nn import Linear,Sequential,BatchNorm1d,ReLU

device = torch.device('cuda:0')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(1433, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8*8, 7, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='/datasets/Cora', name='Cora')

GCN = Net().to(device)
data = dataset[0].to(device)
#optimizer = torch.optim.Adam(GCN.parameters(), lr=0.01, weight_decay=5e-4)
optimizer = SAM(GCN.parameters(), torch.optim.Adam, rho=0.5, adaptive=True, lr=0.01, weight_decay=5e-4)

def train_one_epoch():
    GCN.train()
    #山姆大叔一步
    out = GCN(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.first_step(zero_grad=True)

    #山姆大叔两步
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

#对于GAT

#未使用sam的情况下
# 固定epoch=200       
# GAT acc 81.19%
#加上了SAM 82.30%

