import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from autoencoder import VGAE, VGAE_S
import torch.nn.functional as F 
from torch.distributions.kl import register_kl
from sphere import HypersphericalUniform

#尝试使用S-VAE优化VGAE
print('running S-VAE...')


#变分编码器，实现编码mean和var的任务
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mean = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_var = GCNConv(2 * out_channels, 1, cached=True) 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        mean = self.conv_mean(x,edge_index)
        mean = mean / mean.norm(dim=-1, keepdim=True)
        var = self.conv_var(x,edge_index)
        
        #自增数值1，防止KL崩塌的现象发生
        var = F.softplus(var) + 1 
        return mean,var

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

#定义VGAE_S

model = VGAE_S(VariationalGCNEncoder(num_features, out_channels)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

#TO DO:使用early stop
def train():
    model.train()
    optimizer.zero_grad()
    z,q_z,p_z = model.encode(x, train_pos_edge_index)
    #计算正样例和负样例的交叉熵损失之和
    loss = model.recon_loss(z, train_pos_edge_index)

    #损失加上均值方差的KL散度,使得生成的分布接近于标准正态分布
    q_z.to(device)
    p_z.to(device)
    loss = loss + (1 / data.num_nodes) * model.kl_loss(q_z,p_z)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_one_epoch():
    model.train()
    #SAM首次优化
    z,q_z,p_z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    q_z.to(device)
    p_z.to(device)
    loss = loss + (1 / data.num_nodes) * model.kl_loss(q_z,p_z)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    #SAM二次优化
    z,q_z,p_z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    q_z.to(device)
    p_z.to(device)
    loss = loss + (1 / data.num_nodes) * model.kl_loss(q_z,p_z)
    loss.backward()
    optimizer.second_step(zero_grad=True)
    return loss.item()
    
def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z,mean,var = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
        
for epoch in range(600):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

#VGAE 离SOTA仍然有一些差距
#Epoch: 600, AUC: 0.9056, AP: 0.9156

#更改为S-VGAE之后，发现该方法实际上波动较大,但的确效果较好，参数仍然有优化空间
# Epoch: 599, AUC: 0.9288, AP: 0.9311

