from torch_geometric.nn import LabelPropagation
import torch
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F

device = torch.device('cuda:0')

dataset = Planetoid(root='/datasets/Cora', name='Cora')
data = dataset[0].to(device)

#num_layers控制lp迭代次数 3-6
model = LabelPropagation(num_layers=3, alpha=0.9)
out = model(data.y, data.edge_index, mask=data.train_mask)
pred = out.argmax(dim=1)
correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum()
accuracy = correct / data.test_mask.sum()

print('test acc: ',accuracy.item())

#3 0.603
#4 0.667
#5 0.673
#6 0.69
#7 0.693
#8 0.706
#9 0.707
#15 0.713
#30 0.713

