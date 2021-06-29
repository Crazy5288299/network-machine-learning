import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges
import numpy as np 
import tqdm 
from torch_geometric.datasets.snap_dataset import SNAPDataset

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

dataset = SNAPDataset(root='datasets/ego-facebook', name='ego-facebook', transform=T.NormalizeFeatures())
data = dataset[1]

def show_adj(adj):
    m = adj.shape[0]
    for i in range(m):
        for j in range(m):
            if adj[i,j] == 1:
                print(i,j)

N = len(data.x)
M = data.edge_index.shape[1]
ADJ = torch.zeros((N,N))

edge_index = data.edge_index
file = open('link_res/ego-2-res/-1','w')

for m in range(M):
    u = edge_index[0,m].item()
    v = edge_index[1,m].item()
    file.write(str(u) + ' ' + str(v) + '\n')

file.close()

for m in range(M):
    u = data.edge_index[0,m]
    v = data.edge_index[1,m]
    #print(u,v)
    ADJ[u,v] = 1 
    
#show_adj(ADJ)

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

def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)
    
for epoch in range(50):
    loss = train()
    auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
    #print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

#Epoch: 199, AUC: 0.9114, AP: 0.9128

def get_random_index(m,num_edge=10000):
    rnd_unode = np.random.choice(m,size=num_edge)
    rnd_vnode = np.random.choice(m,size=num_edge)

    idx = np.stack((rnd_unode,rnd_vnode),0)
    return idx 

def adj2edgeindex(adj):
    m = adj.shape[0]
    u_lst = []
    v_lst = []
    for i in range(m):
        for j in range(m):
            if adj[i,j] == 1:
                u_lst.append(i)
                v_lst.append(j)
        idx = np.stack((np.array(u_lst),np.array(v_lst)),0)
    return idx 

#生成链接预测的结果
T = 25
x = x.to(device)
model = model.to(device)

cnt = 0

#最先获取的edge_index
edge_index = edge_index.to(device)

for t in range(T):
    IDX = get_random_index(N)
    IDX = torch.LongTensor(IDX).to(device)

    #使用原来的边进行编码，采用随机的IDX作为解码器
    z = model.encode(x,edge_index)
    p = model.decode(z,IDX)

    #print(p)
    new_node_u = []
    new_node_v = []
    file = open('link_res/ego-2-res/' + str(t), 'w')
    for j,prob in enumerate(p):
        u = IDX[0,j]
        v = IDX[1,j]
        if prob > 0.5:
            if ADJ[u,v] == 0:
                #print('prob',u,v)
                cnt += 1
                ADJ[u,v] = 1
                file.write(str(u.item()) + ' ' +str(v.item()) + '\n')

                new_node_u.append(u.item())
                new_node_v.append(v.item())
        
    new_idx = np.stack((np.array(new_node_u),np.array(new_node_v)),0)
    new_idx = torch.LongTensor(new_idx).to(device)
    edge_index = torch.cat((edge_index,new_idx),1)
    print('edge_index',edge_index.shape)
    file.close()    
    





    


