from .modules import *

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU())
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, num_outputs*dim_output*2))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output, 2)
        return X

class DeepSetOA(nn.Module):
    def __init__(self, dim_input, dim_hidden=128):
        super(DeepSetOA, self).__init__()
        self.dim_hidden = dim_hidden
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU())
        self.weight = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden))
        self.bias = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        weight = self.weight(X).reshape(-1, self.dim_hidden)
        bias = self.bias(X).reshape(-1, self.dim_hidden)
        return weight, bias


class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))
