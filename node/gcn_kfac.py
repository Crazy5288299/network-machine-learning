import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from optim.kfac import KFAC

device = torch.device('cuda:0')

#定义超参数，源代码中使用了网格搜索

EPOCH = 200
GAMMA = 0.5

class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True) 
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

#分为CRD和CLS两步定义GNN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.crd = CRD(1433, 16, 0.5)
        self.cls = CLS(16, 7)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x

dataset = Planetoid(root='/datasets/Cora', name='Cora')

GCN = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(GCN.parameters(), lr=0.01,  weight_decay=5e-4)

#KFAC的超参数

eps = 0.01
update_freq = 50
alpha = None 

#定义KFAC作为预条件器
preconditioner = KFAC(
                GCN, 
                eps, 
                sua=False, 
                pi=False, 
                update_freq=update_freq,
                alpha=alpha if alpha is not None else 1.,
                constraint_norm=False
            )

def train_one_epoch(lam):
    GCN.train()
    optimizer.zero_grad()
    out = GCN(data)
    
    #损失分为有监督和无监督两个部分

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    #loss += lam * F.nll_loss(out[~data.train_mask], data.y[~data.train_mask])

    loss.backward()
    optimizer.step()
    return loss.item()

def test_one_epoch():
    GCN.eval()
    _, pred = GCN(data).max(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
    accuracy = correct / data.test_mask.sum()
    return accuracy.item()

for epoch in range(EPOCH):
    # TO DO:调整GAMMA
    lam = (float(epoch)/float(EPOCH))**GAMMA if GAMMA is not None else 0.
    # lam = 1
    loss = train_one_epoch(lam)
    acc = test_one_epoch()
    preconditioner.step(lam=lam)
    if epoch % 1 == 0:
        print('epoch',epoch,'loss',loss,'accuracy',acc)

#without testset in training
#使用KFAC81.1% 81.7% GAMMA=0.5