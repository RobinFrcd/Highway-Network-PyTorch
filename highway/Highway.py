from typing import Callable

import torch
from torch import nn


class Highway(nn.Module):
    def __init__(
        self, size: int, num_layers: int, activation_fctn: Callable, drop_out: float = 0
    ):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        if drop_out:
            self.nonlinear = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(size, size), nn.Dropout(drop_out))
                    for _ in range(num_layers)
                ]
            )

        else:
            self.nonlinear = nn.ModuleList(
                [nn.Linear(size, size) for _ in range(num_layers)]
            )

        transform_gates = list()
        for _ in range(num_layers):
            transform_gate = nn.Linear(size, size)
            T_gate_bias = (num_layers - 1) // 10 + 1
            transform_gate.bias.data.fill_(-T_gate_bias)
            transform_gates.append(transform_gate)

        self.transform_gate = nn.ModuleList(transform_gates)
        self.activation_fctn = activation_fctn

    def forward(self, x):
        for layer in range(self.num_layers):
            T_gate = torch.sigmoid(self.transform_gate[layer](x))
            H = self.activation_fctn(self.nonlinear[layer](x))

            x = T_gate * H + (1 - T_gate) * x

        return x
