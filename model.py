import torch as t
import torch.nn.functional as F
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.dense import Linear
from torch_geometric.utils import dropout_adj
from torch.nn import Dropout, MaxPool1d
import torch.nn as nn
import torch

from config import config_load

HIDDEN_DIM = 32
LEAKY_SLOPE = 0.2
configs = config_load.get()


def freeze(layer):
    for child in layer.children():
        for p in child.parameters():
            p.requires_grad = False



class MOGT(t.nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim, heads, drop_rate,  edge_dim, residual, devices_available,num_prototypes_per_class):
        super(MOGT, self).__init__()
        self.devices_available = devices_available
        self.drop_rate = drop_rate
        self.convs = t.nn.ModuleList()
        self.residual = residual
        self.laten_dim = int(hidden_dim * heads)
        mid_dim = input_dim + hidden_dim  if residual else hidden_dim

        self.convs.append(TransformerConv(input_dim, hidden_dim, heads=heads,  edge_dim=edge_dim,
                                          concat=False, beta=True).to(self.devices_available))
        self.convs.append(TransformerConv(mid_dim, hidden_dim, heads=heads,
                                           edge_dim=edge_dim, concat=True, beta=True).to(self.devices_available))
        
        self.ln1 = LayerNorm(in_channels=mid_dim).to(self.devices_available)
        self.ln2 = LayerNorm(in_channels=hidden_dim *
                             heads).to(self.devices_available)
        
        self.pool = MaxPool1d(2, 2)

        # prototype layers
        self.epsilon = 1e-4         
        self.prototype_shape = (output_dim * num_prototypes_per_class, self.laten_dim)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True).to(self.devices_available)
        self.num_prototypes = self.prototype_shape[0]

        self.dropout = Dropout(drop_rate)
        self.lins = t.nn.ModuleList()
        self.lins.append(
            Linear(int(hidden_dim*heads /2), HIDDEN_DIM, weight_initializer="kaiming_uniform").to(devices_available))
        self.lins.append(
            Linear(HIDDEN_DIM, 1, weight_initializer="kaiming_uniform").to(devices_available))

        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    output_dim)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
    
    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors)) 
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True)) 
        similarity = torch.log((distance + 1) / (distance + self.epsilon)) 
        return similarity, distance
        

    def forward(self, data):
        data = data.to(self.devices_available)
        x = data.x
        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        res = x
        x = self.convs[0](x, edge_index, edge_attr)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE, inplace=True)
        x = t.cat((x, res), dim=1) if self.residual else x
        x = self.ln1(x)

        edge_index, edge_attr = dropout_adj(data.edge_index, data.edge_attr, p=self.drop_rate, force_undirected=True,
                                            training=self.training)
        x = self.convs[1](x.to(self.devices_available), edge_index.to(
            self.devices_available), edge_attr.to(self.devices_available))
        x = self.ln2(x)
        x = F.leaky_relu(x, negative_slope=LEAKY_SLOPE)

        graph_emb = x
        x = t.unsqueeze(x, 1)
        x = self.pool(x)
        x = t.squeeze(x) 

        #prototype
        prototype_activations, min_distance = self.prototype_distances(graph_emb)

        final_embedding = torch.cat((prototype_activations, graph_emb), dim=1)
        
        x = self.lins[0](x).relu()
        x = self.dropout(x)
        x = self.lins[1](x)
        probs = t.sigmoid(x)

        KL_Loss = 0

        return probs,prototype_activations,KL_Loss

    def gumbel_softmax(self, prob):
        return F.gumbel_softmax(prob, tau = 1, dim = -1)




class FocalLoss(t.nn.Module):
    def __init__(self, alpha=0.25, gamma=2,logits=False, reduction=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.logits = logits

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = t.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction:
            return t.mean(F_loss)
        else:
            return F_loss