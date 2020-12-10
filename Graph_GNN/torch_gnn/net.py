import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes: typing.Iterable[int], out_dim, activation_function=nn.Sigmoid(),
                 activation_out=None):
        super(MLP, self).__init__()

        i_h_sizes = [input_dim] + hidden_sizes  # add input dim to the iterable
        self.mlp = nn.Sequential()
        for idx in range(len(i_h_sizes) - 1):
            self.mlp.add_module("layer_{}".format(idx),
                                nn.Linear(in_features=i_h_sizes[idx], out_features=i_h_sizes[idx + 1]))
            self.mlp.add_module("act", activation_function)
        self.mlp.add_module("out_layer", nn.Linear(i_h_sizes[-1], out_dim))
        if activation_out is not None:
            self.mlp.add_module("out_layer_activation", activation_out)

    def init(self):
        for i, l in enumerate(self.mlp):
            if type(l) == nn.Linear:
                nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        return self.mlp(x)


# code from Pedro H. Avelar

class StateTransition(nn.Module):

    def __init__(self,
                 node_state_dim: int,
                 node_label_dim: int,
                 mlp_hidden_dim: typing.Iterable[int],
                 activation_function=nn.Tanh()
                 ):
        super(type(self), self).__init__()
        d_i = node_state_dim + 2 * node_label_dim  # arc state computation f(l_v, l_n, x_n)
        d_o = node_state_dim
        d_h = list(mlp_hidden_dim)  # if already a list, no change
        self.mlp = MLP(input_dim=d_i, hidden_sizes=d_h, out_dim=d_o, activation_function=activation_function,
                       activation_out=activation_function)  # state transition function, non-linearity also in output

    def forward(
            self,
            node_states,
            node_labels,
            edges,
            agg_matrix,
    ):
        src_label = node_labels[edges[:, 0]]
        tgt_label = node_labels[edges[:, 1]]
        tgt_state = node_states[edges[:, 1]]
        edge_states = self.mlp(
            torch.cat(
                [src_label, tgt_label, tgt_state],
                -1
            )
        )

        new_state = torch.matmul(agg_matrix, edge_states)
        return new_state
