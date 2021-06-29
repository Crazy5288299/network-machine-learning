import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges
import numpy as np 

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

dataset = Planetoid(root='/datasets/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]

def show_adj(adj):
    m = adj.shape[0]
    for i in range(m):
        for j in range(m):
            if adj[i,j] == 1:
                print(i,j)

N = len(data.x)
M = data.edge_index.shape[1]
ADJ = torch.zeros((N,N))

for m in range(M):
    u = data.edge_index[0,m]
    v = data.edge_index[1,m]
    #print(u,v)
    ADJ[u,v] = 1 

show_adj(ADJ)

#先去除原数据中的mask，再用负采样边分割训练集和测试集
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)

out_channels = 16
num_features = dataset.num_features

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

#定义endoder, decoder GAE默认为内积相似度计算, VGAE则直接由生成的正态分布中采样

model = GAE(GCNEncoder(num_features, out_channels)).to(device)
#model = GAE(LinearEncoder(num_features, out_channels)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    #计算正样例和负样例的交叉熵损失之和
    loss = model.recon_loss(z, train_pos_edge_index)

    #VGAE加上该行，损失计算均值方差的KL散度
    #loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

def train_one_epoch():
    model.train()
    #山姆大叔一步
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.first_step(zero_grad=True)

    #山姆大叔两步
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.second_step(zero_grad=True)
    return loss.item()

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
    
for epoch in range(200):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

#Epoch: 199, AUC: 0.9114, AP: 0.9128