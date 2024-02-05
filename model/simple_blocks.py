import torch
import torch.nn as nn
from model.primitives import Linear, LayerNorm


    
class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = Linear(self.c, self.c, init='relu')
        self.linear_2 = Linear(self.c, self.c, init='relu')
        self.linear_3 = Linear(self.c, self.c, init='final')

        self.relu = nn.LeakyReLU(0.5)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s


class StructureModuleTransition(nn.Module):
    def __init__(self, c, num_layers=2, dropout_rate=0.1):
        super(StructureModuleTransition, self).__init__()

        self.c = c
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            l = StructureModuleTransitionLayer(self.c)
            self.layers.append(l)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        for l in self.layers:
            s = l(s)

        #s = self.dropout(s)
        s = self.layer_norm(s)

        return s


class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s, n=8):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        '''self.linear1 = Linear(self.c_s, n*self.c_s)
        self.linear2 = Linear(n*self.c_s, 6, init='final')'''
        self.linear = Linear(self.c_s, 6, init='final')

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector 
        """
        # [*, 6]
        #update = self.linear2(self.linear1(s))
        update = self.linear(s)

        return update