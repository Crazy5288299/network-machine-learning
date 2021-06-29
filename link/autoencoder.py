import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from ive import ive 
from vmf import VonMisesFisher
from sphere import HypersphericalUniform
from torch.distributions.kl import register_kl

EPS = 1e-15
MAX_LOGSTD = 10

#重置模型参数
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

#基于向量内积的解码器
class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

#图自编码器，用encoder初始化，decoder默认为基于向量内积的解码器
class GAE(torch.nn.Module):
    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss
    
    #计算评价指标
    def test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)


class VGAE(GAE):
    def __init__(self, encoder, decoder=None):
        super(VGAE, self).__init__(encoder, decoder)

    #重参数化技术
    def reparametrize(self, mu, logstd):
        if self.training:
            #训练过程中根据mu和logstd计算新的mu
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    #计算正态分布的KL散度
    def kl_loss(self, mu=None, logstd=None):
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        #实际上返回负的KL散度
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))


#对抗正则自编码器
class ARGA(GAE):
    def __init__(self, encoder, discriminator, decoder=None):
        super(ARGA, self).__init__(encoder, decoder)
        self.discriminator = discriminator
        reset(self.discriminator)

    def reset_parameters(self):
        super(ARGA, self).reset_parameters()
        reset(self.discriminator)

    def reg_loss(self, z):
        real = torch.sigmoid(self.discriminator(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss

    #计算判别器的损失
    def discriminator_loss(self, z):
        #假设真实分布满足正态分布
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss

#对抗正则变分自编码器
class ARGVA(ARGA):
    def __init__(self, encoder, discriminator, decoder=None):
        super(ARGVA, self).__init__(encoder, discriminator, decoder)
        self.VGAE = VGAE(encoder, decoder)

    @property
    def __mu__(self):
        return self.VGAE.__mu__

    @property
    def __logstd__(self):
        return self.VGAE.__logstd__

    def reparametrize(self, mu, logstd):
        return self.VGAE.reparametrize(mu, logstd)

    def encode(self, *args, **kwargs):
        """"""
        return self.VGAE.encode(*args, **kwargs)

    def kl_loss(self, mu=None, logstd=None):
        return self.VGAE.kl_loss(mu, logstd)

#TO DO: GAN -> W-GAN, VAE -> S-VAE

#用WGAN更改对抗正则自编码器
class ARGA_WGAN(ARGVA):
    def __init__(self, encoder, discriminator, decoder=None,clip_value=0.01):
        super().__init__(encoder, discriminator, decoder)
        self.VGAE = VGAE(encoder, decoder)
        self.clip_value = clip_value
        self.discriminator = discriminator

    #对判别器的损失进行修改
    def discriminator_loss(self, z):
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -real.mean()
        fake_loss = -(1 - fake).mean()
        return real_loss + fake_loss

    #裁剪weight，使得权重满足liptiz条件
    def clip_weight(self):
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)

#使用S-VAE代替VAE

#注册kl散度
@register_kl(VonMisesFisher, HypersphericalUniform)
def _kl_vmf_uniform(vmf, hyu):
    return -vmf.entropy() + hyu.entropy()
        
class VGAE_S(GAE):
    def __init__(self, encoder, decoder=None):
        super().__init__(encoder, decoder)

    #重参数化技术
    def reparametrize(self, z_mean, z_var):
        q_z = VonMisesFisher(z_mean, z_var)
        z_dim = z_mean.shape[1]
        p_z = HypersphericalUniform(z_dim - 1)
        return q_z, p_z

    def encode(self, *args, **kwargs):
        z_mean,z_var = self.encoder(*args, **kwargs)
        q_z, p_z = self.reparametrize(z_mean,z_var)
        z = q_z.rsample()
        return  z,q_z, p_z

    #计算vmf分布的负KL散度
    def kl_loss(self,q_z,p_z):
        loss_KL = -torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        return loss_KL
