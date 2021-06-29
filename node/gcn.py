import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv,SAGEConv
from torch.nn import Linear,Sequential,BatchNorm1d,ReLU
device = torch.device('cuda:0')

class Net(torch.nn.Module):
    def __init__(self,dim=16):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1433, 16)
        self.conv2 = GCNEConv(16, 7)

        #可以选用图同构神经网络GIN
        """
        self.conv1 = GINConv(
            Sequential(Linear(1433, dim), BatchNorm1d(dim), ReLU(),
                       Linear(dim, dim), ReLU()))

        self.conv2 = GINConv(Linear(dim,7))
        """
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

dataset = Planetoid(root='/datasets/Cora', name='Cora')

GCN = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(GCN.parameters(), lr=0.01, weight_decay=5e-4)

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
for epoch in range(200):
    loss = train_one_epoch()
    acc = test_one_epoch()
    if epoch % 1 == 0:
        print('epoch',epoch,'loss',loss,'accuracy',acc)


# 固定epoch=200       
# GCN acc 81.10%