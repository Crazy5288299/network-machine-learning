import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from autoencoder import VGAE

#使用基于GCN的图变分自编码器进行链接预测

#变分编码器，实现编码mu和log_std的任务
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

print('runnign VGAE')

dataset = Planetoid(root='/datasets/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]
#先去除原数据中的mask，再用负采样边分割训练集和测试集
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data)

out_channels = 16
num_features = dataset.num_features

#负采样
device = torch.device('cuda:1')
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)

#定义endoder, VGAE则直接由生成的正态分布中采样

model = VGAE(VariationalGCNEncoder(num_features, out_channels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    #计算正样例和负样例的交叉熵损失之和
    loss = model.recon_loss(z, train_pos_edge_index)

    #损失加上均值方差的KL散度,使得生成的分布接近于标准正态分布
    loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return loss.item()

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
        
for epoch in range(600):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

#VGAE 离SOTA仍然有一些差距
#Epoch: 400, AUC: 0.8979, AP: 0.9068
#Epoch: 600, AUC: 0.9056, AP: 0.9156

