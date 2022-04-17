import DendriticLayer
import Classifiers as cl
import torch
import torch.nn as nn

# TODO: 2 CNN layers or 3? 8x8 or 4x4? how many filters? how to map multiple filters to a classifier?


class CNN(nn.Module):
    def __init__(self, num_convs: int = 3, kernel: int = 5, num_filters: int = 4, num_in_channels: int = 3):
        super().__init__()

        layers = []
        in_channels = num_in_channels
        filters = num_filters

        for i in list(range(num_convs)):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=filters, 
                          kernel_size=kernel, stride=2, padding=1),
                nn.BatchNorm2d(num_features=filters),
                nn.ReLU()
            ))
            in_channels = filters
            filters = 2 * filters

        self.conv_layers = nn.Sequential(*layers)


    def forward(self, x):
        return self.conv_layers(x)



class SkipMLN(nn.Module):

    def __init__(self, in_features: int, num_layers: int = 5, **kwargs):
        """
        * `in_features` - size of the input sample
        * `num_classes` - number of classes to predict (number of neurons)
        Use `**kwargs` to pass desired parameters to each dendritic layer
        """
        super(SkipMLN, self).__init__()
        self.num_neurons = in_features
        self.neuron_layers = []
        for i in list(range(num_layers)):
            self.neuron_layers.append(DendriticLayer(in_features ** 2, **kwargs))

    def forward(self, x: torch.Tensor):
        inputs = x
        for i in list(range(self.num_neurons)):
            z = self.neuron_layers[i](inputs.repeat(1, self.num_neurons))
            inputs = x + z
        return inputs


num_convs = 3
class ConvMLNNet(nn.Module):

    def __init__(self, image_size: int, num_classes: int, conv_args: dict = {}, neuron_args: dict = {}):
        """
        * `image_size` - size of the image (one dimensions, assumed to be a square)
        * `num_classes` - number of classes to predict (number of neurons)
        * `conv_args` - additional parameters for the convolutional neural net (`num_convs`, `num_in_channels`, `num_filters`, `kernel`)
        * `neuron_args` - additional parameters for the dendritic neurons (`branching`, `bias`, `dropout`, `activation`)
        """
        self.conv_layers = CNN(**conv_args)
        num_conv_layers = conv_args['num_convs'] if 'num_convs' in conv_args else 3
        self.classifier = cl.MLNClassifier(num_classes * (image_size / 2^num_conv_layers)**2, num_classes, **neuron_args)
        self.num_neurons = num_classes

    def forward(self, x: torch.Tensor):
        z = self.conv_layers(x)
        # TODO: flatten
        y = self.classifier(z.repeat(1, self.num_neurons))
        return y



num_convs = 2
class ConvDendriteNet(nn.Module):

    def __init__(self, image_size: int, num_classes: int, conv_args: dict = {}, reduce_args: dict = {}, classifier_args: dict = {}):
        """
        * `image_size` - size of the image (one dimensions, assumed to be a square)
        * `num_classes` - number of classes to predict (number of neurons)
        * `conv_args` - additional parameters for the convolutional neural net (`num_convs`, `num_in_channels`, `num_filters`, `kernel`)
        * `neuron_args` - additional parameters for the dendritic neurons (`branching`, `bias`, `dropout`, `activation`)
        """
        self.conv_layers = CNN(**conv_args)
        num_conv_layers = conv_args['num_convs'] if 'num_convs' in conv_args else 3
        num_filters = conv_args['num_filters'] if 'num_filters' in conv_args else 4
        conv_output_size = image_size / 2^num_conv_layers
        conv_output_filters = num_filters * 2^(num_conv_layers-1)

        self.reduce_layer = DendriticLayer(in_features=conv_output_size * conv_output_filters, 
                                           out_features=conv_output_filters, **reduce_args)
        self.classifier = cl.MLNClassifier(num_classes * conv_output_filters, num_classes, **classifier_args)
        self.num_neurons = num_classes

    def forward(self, x: torch.Tensor):
        z = self.conv_layers(x)
        # TODO: flatten
        y = self.reduce_layer(z)
        return self.classifier(y)


num_in_channels = 1
num_filters = 4
kernel = 5