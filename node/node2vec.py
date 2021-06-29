import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
#from torch_geometric.nn import Node2Vec
from mynode2vec import Node2Vec

def main():
    dataset = Planetoid(root='/datasets/Cora', name='Cora')
    data = dataset[0]
    device = torch.device('cuda')

    #p,q表示BFS,DFS的不同采样概率
    model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                     context_size=10, walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        #使用torch Embedding 对随机游走的结果做嵌入，采用逻辑斯特回归做分类

        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            #loss 正负随机游走的embedding的损失之和，希望正样例之间的对数似然更大，负样例之间的更小
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         atype = 'Forest',
                         max_iter=100)
        return acc

    for epoch in range(200):
        loss = train()
        acc = test()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.4f}')

    @torch.no_grad()
    def plot_points(colors):
        model.eval()
        z = model(torch.arange(data.num_nodes, device=device))
        #可视化聚类结果
        z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
        y = data.y.cpu().numpy()

        plt.figure(figsize=(8, 8))
        for i in range(dataset.num_classes):
            plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
        plt.axis('off')
        plt.savefig("./images/node2vec.png")
        plt.show()
        

    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]
    #plot_points(colors)


if __name__ == "__main__":
    main()

#Logistic 
#50轮次准确率达到68.7%
#200轮次准确率达到72.7%

#SVM
#200 epoch: 75.2%

#Decision Tree
#200 epoch: 43.5%

#Forest
#200 epoch: 69.7%