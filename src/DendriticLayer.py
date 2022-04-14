import math
import numpy as np
import torch
import torch.nn as nn

class DendriticLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int = None, 
                 branching: int = 2, depth: int = None, 
                 bias: bool = True, dropout: float = 0.1,
                 activation = nn.LeakyReLU, **kwargs):
        """
        * `in_features` - size of the input sample ($I$)
        * `out_features` - size of the output sample ($O$), AKA the number of neurons in the layer
        * `branching` - the branching factor for the dendritic tree ($b$)
        * `depth` - the depth of each dendritic tree ($d$)
        * `bias` - whether to include the bias term in the linear layers of the denderite trees. Default=False (not sure if this is possible with the mask??)
        * `dropout` - where to include dropout in the trees (??)
        * `activation` - the activation function to use between layers of the tree (e.g. nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.tanh, etc.);
        pass `None` if no activation between layers is desired. Default = nn.LeakyReLU
        Use `**kwargs` to pass desired parameters to the activation layers. Default = negative_slope=0.1 for nn.LeakyReLU

        The following relationship must hold:
        $$ in_features = out_features \times (branching)^{depth}$$
        where $depth$ is the number of layers in each dendritic neuron.

        Users should specify either `out_features` or `depth`
        The other one is determined automatically
        """
        super(DendriticLayer, self).__init__()

        if isinstance(out_features, type(None)):
          # Determine the number of neurons based on the branching & depth
          out_features = int(in_features / (branching ** depth))
          assert out_features % 1 == 0
        else:
          # Determine the depth based on the number of neurons & branching
          assert in_features % out_features == 0
          depth = math.log(in_features / out_features, branching)
          assert depth % 1 == 0

        self.depth = int(depth)
        self.bias = bias

        # Check that the activation function is either None or comes from the torch activation module
        if not isinstance(activation, type(None)):
          assert activation.__name__ in dir(nn.modules.activation)
        
        # If the activation function was left to the default LeakyReLU but no kwargs were given,
        # set the default negative slope to 0.1
        if activation == nn.LeakyReLU and len(kwargs.keys()) == 0:
          kwargs['negative_slope'] = 0.1

        # Initialize a weight matrix for each level of the tree
        # and simultaneously create a mask for each level to enforce tree structure
        weights = []
        self.masks = []
        activations = []
        num_in = in_features
        for i in list(range(self.depth)):
          num_out = int(num_in / branching)

          # Use Kaiming He initialization to account for nonlinear activations and sparsity
          # TODO: double check this code
          weight_layer = nn.Linear(num_in, num_out, bias=bias)
          density = num_in / (num_in * num_out)   # sparsity compared to full MLP
          weight_layer.weight.data = torch.nn.init.normal_(weight_layer.weight.data, mean=0.0, std=math.sqrt(2/(num_in*density)))
          weights.append(weight_layer)

          self.masks.append(self.build_mask(num_in, branching))
          if not isinstance(activation, type(None)):
            activations.append(activation(**kwargs))
          num_in = num_out

        self.weights = nn.ModuleList(weights)
        self.activations = nn.ModuleList(activations)

        # optional dropout layer; implement later once you understand better
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x: torch.Tensor):
      x = self.dropout(x)

      for i in list(range(self.depth)):
        masked_weight = (self.weights[i].weight * self.masks[i]).float()
        x = torch.mm(x, torch.t(masked_weight))
        if self.bias:
          x = x + self.weights[i].bias
        if (len(self.activations) > 0):
          x = self.activations[i](x)

      return x
      

    def build_mask(self, in_features: int, branching: int):
      dim1 = np.repeat(list(range(int(in_features / branching))), branching)
      dim2 = list(range(in_features))

      mask = np.zeros((in_features // branching, in_features))
      mask[dim1, dim2] = 1
      return(torch.tensor(mask))
