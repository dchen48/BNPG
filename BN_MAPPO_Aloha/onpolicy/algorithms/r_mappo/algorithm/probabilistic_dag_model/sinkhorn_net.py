
"""Model class for sorting numbers."""

import torch.nn as nn

class Sinkhorn_Net(nn.Module):

    def __init__(self, input_dim, latent_dim, output_dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        in_flattened_vector: input flattened vector
        latent_dim: number of neurons in latent layer
        output_dim: dimension of log alpha square matrix
        """
        super(Sinkhorn_Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear1 = nn.Linear(self.input_dim, latent_dim)
        self.relu1 = nn.ReLU()
        # now those latent representation are connected to rows of the matrix
        # log_alpha.
        self.linear2 = nn.Linear(latent_dim, output_dim*output_dim)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        """
        # each number is processed with the same network, so data is reshaped
        # so that numbers occupy the 'batch' position.
 
        # activation_fn: ReLU
        x = self.relu1(self.linear1(x))
        # no activation function is enabled
        x = self.linear2(x)
        #reshape to cubic for sinkhorn operation
        x = x.view(-1, self.output_dim, self.output_dim)
        return x
