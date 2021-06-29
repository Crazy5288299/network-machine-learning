import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score, homogeneity_score, completeness_score
#聚类效果指标，用同质性与完整性衡量 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from autoencoder import ARGA_WGAN
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges

dataset = Planetoid("/datasets/Cora", "Cora", transform=T.NormalizeFeatures())
data = dataset[0]

data.train_mask = data.val_mask = data.test_mask = None
data = train_test_split_edges(data)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv_mu = GCNConv(hidden_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Discriminator(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x

#参数设置和论文略有不同
encoder = Encoder(data.num_features, hidden_channels=32, out_channels=32)
discriminator = Discriminator(in_channels=32, hidden_channels=64, out_channels=32)
model = ARGA_WGAN(encoder, discriminator,clip_value=0.01)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model, data = model.to(device), data.to(device)

#优化器从Adam改为RMSprop
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),lr=0.01)
encoder_optimizer = torch.optim.RMSprop(encoder.parameters(), lr=0.005)

def train():
    model.train()
    encoder_optimizer.zero_grad()
    #使用变分生成器生成出尽量逼真的负样例
    z = model.encode(data.x, data.train_pos_edge_index)
    #print(z)

    for i in range(5):
        discriminator.train()
        discriminator_optimizer.zero_grad()
        discriminator_loss = model.discriminator_loss(z)
        #鉴别器的损失包含正样例和负样例的损失
        discriminator_loss.backward()
        discriminator_optimizer.step()

        #对鉴别器进行权重裁剪
        model.clip_weight()

    loss = model.recon_loss(z, data.train_pos_edge_index)
    loss = loss + (1 / data.num_nodes) * model.kl_loss()

    loss.backward()
    encoder_optimizer.step()
    return loss

@torch.no_grad()
def test():
    model.eval()
    z = model.encode(data.x, data.train_pos_edge_index)

    #用Kmeans进行7个类别的聚类
    kmeans_input = z.cpu().numpy()
    kmeans = KMeans(n_clusters=7, random_state=0).fit(kmeans_input)
    pred = kmeans.predict(kmeans_input)

    #用聚类的指标评判编码器的训练效果
    
    labels = data.y.cpu().numpy()
    completeness = completeness_score(labels, pred)
    hm = homogeneity_score(labels, pred)
    nmi = v_measure_score(labels, pred)
    
    #计算链接预测的准确率
    auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
    
    return auc, ap, completeness, hm, nmi

#TO DO: 显式使用early stop
for epoch in range(200):
    loss = train()
    auc, ap, completeness, hm, nmi = test()
    print((f'Epoch: {epoch:03d}, Loss: {loss:.3f}, AUC: {auc:.3f}, AP: {ap:.3f}, NMI: {nmi:.3f}'))
    #print((f'Completeness: {completeness:.3f}, 'f'Homogeneity: {hm:.3f}, NMI: {nmi:.3f}'))

#从GAN到WGAN,好像并没有做出效果的显著提升...
# GAN:Epoch: 199, Loss: 0.872, AUC: 0.925, AP: 0.929

# WGAN Epoch: 499, Loss: 0.942, AUC: 0.909, AP: 0.901, NMI: 0.500
#Epoch: 1999, Loss: 0.865, AUC: 0.913, AP: 0.929, NMI: 0.447