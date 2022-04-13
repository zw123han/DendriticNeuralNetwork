class MLNBinaryClassifier(nn.Module):

    def __init__(self, in_features: int, **kwargs):
        """
        * `in_features` - size of the input sample
        Use `**kwargs` to pass desired parameters to the dendritic layer
        """
        super(MLNBinaryClassifier, self).__init__()
        self.neuron = DendriticLayer(in_features, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        z = self.neuron(x)
        y = self.sigmoid(z)
        return y


class MLNClassifier(nn.Module):

    def __init__(self, in_features: int, num_classes: int = None, **kwargs):
        """
        * `in_features` - size of the input sample
        * `num_classes` - number of classes to predict (number of neurons)
        Use `**kwargs` to pass desired parameters to the dendritic layer
        """
        super(MLNClassifier, self).__init__()
        self.num_neurons = num_classes
        self.neurons = DendriticLayer(in_features * num_classes, **kwargs)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        z = self.neurons(x.repeat(1, self.num_neurons))
        y = self.softmax(z)
        return y 