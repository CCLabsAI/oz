import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Dropout

import oz


class BasicMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepFullyConnected(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        last_size = input_size
        self.fc = nn.ModuleList()
        for size in hidden_sizes:
            self.fc.append(nn.Linear(last_size, size))
            last_size = size
        self.fc_logit = nn.Linear(last_size, output_size)

    def forward(self, x):
        for fc in self.fc:
            x = F.relu(fc(x))
        x = self.fc_logit(x)
        return x


def build_holdem_demo_model():
    return torch.nn.Sequential(
        Dropout(p=0.1),
        Linear(in_features=(2+5)*(13+4)+(2*4*6*2), out_features=1024),
        ReLU(),
        Linear(in_features=1024, out_features=512),
        ReLU(),
        Dropout(p=0.5),
        Linear(in_features=512, out_features=1024),
        ReLU(),
        Linear(in_features=1024, out_features=512),
        ReLU(),
        Dropout(p=0.5),
        Linear(in_features=512, out_features=3)
    )


def model_with_args(args, input_size, output_size):
    nn_arch = args.nn_arch

    if nn_arch == 'mlp':
        hidden_size = args.hidden_size
        return BasicMLP(input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size)
    elif nn_arch == 'deep':
        hidden_size_str = args.hidden_sizes
        hidden_sizes = [int(size) for size in hidden_size_str.split(':')]
        return DeepFullyConnected(input_size=input_size,
                                  hidden_sizes=hidden_sizes,
                                  output_size=output_size)
    elif nn_arch == 'holdem_demo':
        return build_holdem_demo_model()
    else:
        raise 'unknown NN architecture'
