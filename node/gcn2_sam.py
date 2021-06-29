import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv,SAGEConv
from torch.nn import Linear,Sequential,BatchNorm1d,ReLU
from torch_geometric.nn import GCN2Conv
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from optim.sam import SAM

device = torch.device('cuda:0')

#添加边特征等输入GCNII训练
transform = T.Compose([T.NormalizeFeatures(),T.ToSparseTensor()])
dataset = Planetoid(root='/datasets/Cora', name='Cora',transform=transform)

#使用GCNII进行实验

class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0):
        super(Net, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))

        self.dropout = dropout

    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1)

#使用64层GCN
GCN = Net(hidden_channels=64, num_layers=64, alpha=0.1, theta=0.5,
            shared_weights=True, dropout=0.6).to(device)


data = dataset[0].to(device)
data.adj_t = gcn_norm(data.adj_t)  # GCN标准化作为预处理手段

params_to_optim = [
    dict(params=GCN.convs.parameters(), weight_decay=0.01),
    dict(params=GCN.lins.parameters(), weight_decay=5e-4)
]

optimizer = SAM(params_to_optim, torch.optim.Adam, rho=0.5, adaptive=True, lr=5e-3)

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

GCN.train()
best_acc = 0
for epoch in range(1000):
    loss = train_one_epoch()
    acc = test_one_epoch()
    if acc > best_acc:
        best_acc = acc
    if epoch % 1 == 0:
        print('epoch',epoch,'loss',loss,'accuracy',acc)
print('best accuracy',best_acc)

# GCNII acc 85.50%

# GCNII+SAM 并且使用网格搜索策略
# 5e-3 86.50%
# 4e-3 85.50%
# 3e-3 85.60%
# 2e-3 83.80%