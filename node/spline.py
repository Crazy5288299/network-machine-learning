import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import SplineConv
import torch_geometric.transforms as T

#样条卷积

device = torch.device('cuda:2')

#官方代码中更改了数据集

#预先保留全局归一化度数信息
transform = T.Compose([
    T.TargetIndegree(),
])
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(1433, 16, dim=1, kernel_size=2)
        self.conv2 = SplineConv(16, 7, dim=1, kernel_size=2)

    def forward(self, data):
        x, edge_index,edge_attr  = data.x, data.edge_index,data.edge_attr
        #print(edge_attr)
        x = F.dropout(x, training=self.training)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='/datasets/Cora', name='Cora', transform=transform)

GCN = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(GCN.parameters(), lr=0.01, weight_decay=5e-3)

def train_one_epoch():
    GCN.train()
    optimizer.zero_grad()
    out = GCN(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test_one_epoch():
    GCN.eval()
    _, pred = GCN(data).max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
    accuracy = correct / data.test_mask.sum()
    return accuracy.item()

GCN.train()
for epoch in range(100):
    loss = train_one_epoch()
    acc = test_one_epoch()
    if epoch % 1 == 0:
        print('epoch',epoch,'loss',loss,'accuracy',acc)

# 固定epoch=200       
# GCN acc 81.49%