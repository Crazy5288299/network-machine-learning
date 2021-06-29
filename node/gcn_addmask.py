import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv
from torch_geometric.nn import GCNConv
from optim.sam import SAM

transform = T.Compose([
    T.AddTrainValTestMask('train_rest', num_val=500, num_test=500),
    T.TargetIndegree(),
])
path = '/datasets/Cora'
dataset = Planetoid(path, 'Cora', transform=transform)
data = dataset[0]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = SplineConv(dataset.num_features, 16, dim=1, kernel_size=2)
        #self.conv2 = SplineConv(16, dataset.num_classes, dim=1, kernel_size=2)
        
        self.conv1 = GCNConv(1433, 16)
        self.conv2 = GCNConv(16, 7)
    def forward(self):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.dropout(x, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    log_probs, accs = model(), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = log_probs[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))