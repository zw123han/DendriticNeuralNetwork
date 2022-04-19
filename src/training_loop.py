
import math
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

# To filter mnist digits: https://stackoverflow.com/questions/57913825/how-to-select-specific-labels-in-pytorch-mnist-dataset
class YourSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, mask, data_source):
        self.mask = mask
        self.data_source = data_source

    def __iter__(self):
        return iter([i.item() for i in torch.nonzero(mask)])

    def __len__(self):
        return len(self.data_source)

# run this to convert mnist 4/9 labels to 1 or 0
def convert_bool(y):
  return (y == 9).float()
  

# Load different datasets based on args.
def load_datasets(args):
  use_cuda = True
  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

  if(args["dataset"] == "mnist"):
    mnist_train = datasets.MNIST('data', train=True, download=True,
                          transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,),(0.3081,)),
                        ]))


    mnist_validation = datasets.MNIST('data', train=False, download=True, 
                                transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,),(0.3081,)),
                        ]))

    mnist_test = datasets.MNIST('data', train=False, download=True, 
                                transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,),(0.3081,)),
                        ]))


    bs = 128
    train_dl = DataLoader(mnist_train, batch_size=bs, shuffle=True, **kwargs)


    validation_dl = DataLoader(mnist_validation, batch_size = 100, shuffle=True, **kwargs)

    test_dl = DataLoader(mnist_test, batch_size = 100, shuffle=True, **kwargs)

    dataiter = iter(train_dl)
    train_x, train_y = dataiter.next()
    train_x = nn.Upsample(size=(32, 32))(train_x)



    dataiter = iter(validation_dl)
    validation_x, validation_y = dataiter.next()
    validation_x = nn.Upsample(size=(32, 32))(validation_x)


    dataiter = iter(test_dl)
    test_x, test_y = dataiter.next()
    test_x = nn.Upsample(size=(32, 32))(test_x)


  elif(args["dataset"] == "mnist-49"):

    mnist_train = datasets.MNIST('data', train=True, download=True,
                          transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,),(0.3081,)),
                        ]))

    bs = 128
    train_dl = DataLoader(mnist_train, batch_size=len(mnist_train))
    dataiter = iter(train_dl)
    train_x, train_y = dataiter.next()
    train_x = nn.Upsample(size=(32, 32))(train_x)
    filted_indices = np.logical_or(train_y == 4, train_y == 9)
    train_y = train_y[filted_indices]
    train_x = train_x[filted_indices]
    sampled_indics = np.random.randint(0, len(train_x), size = bs)
    train_y = convert_bool(train_y[sampled_indics]).unsqueeze(dim=-1)
    train_x = train_x[sampled_indics]

    # Validation and Test
    mnist_validation_test = datasets.MNIST('data', train=False, download=True,
                          transform = torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,),(0.3081,)),
                        ]))

    bs = 100
    validation_test_dl = DataLoader(mnist_train, batch_size=len(mnist_validation_test))
    dataiter = iter(validation_test_dl)
    validation_test_x, validation_test_y = dataiter.next()
    validation_test_x = nn.Upsample(size=(32, 32))(validation_test_x)

    # Validation
    filted_indices = np.logical_or(validation_test_y == 4, validation_test_y == 9)
    validation_test_y = validation_test_y[filted_indices]
    validation_test_x = validation_test_x[filted_indices]
    validation_sampled_indics = np.random.randint(0, len(validation_test_x), size = bs)
    validation_y = convert_bool(validation_test_y[validation_sampled_indics]).unsqueeze(dim=-1)
    validation_x = validation_test_x[validation_sampled_indics]



    # Test
    filted_indices = np.logical_or(validation_test_y == 4, validation_test_y == 9)
    test_sampled_indics = np.random.randint(0, len(validation_test_x), size = bs)
    test_y = convert_bool(validation_test_y[test_sampled_indics]).unsqueeze(dim=-1)
    test_x = validation_test_x[test_sampled_indics]

  return train_x.view(-1, 32*32).to(device), validation_x.view(-1, 32*32).to(device), test_x.view(-1, 32*32).to(device), train_y.to(device), validation_y.to(device), test_y.to(device)


def accuracy(output, labels):

    # From programming assignment 4.
    predictions = output.max(1)[1].type_as(labels)
    correct = predictions.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def accuracy(output, labels, binary = False):

    # From programming assignment 4.
    if binary:
      return torch.mean(((output > 0.5) == labels.bool()).float())
    predictions = output.max(1)[1].type_as(labels)
    correct = predictions.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train(model, args):
  # model is the model we are going to train.
  # args is a dictionary of all the hyperparameters.

  train_losses = []
  valid_losses = []
  valid_accs = []
  # Load optimizer, and loss.
  #optimizer = args['optimizer']() not using function pointer
  #loss = args['loss']()


  # train_x, validation_x, test_x, train_y, validation_y, test_y = load_datasets(args)
  binary = False # for accuracy function
  if args["optimizer"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])

  if args["criterion"] == "cross_entropy":
    loss = nn.CrossEntropyLoss()
  
  if args["criterion"] == "bce":
    loss = nn.BCELoss()
    binary = True

  model = model.to(device=device)
  for epoch in range(args["epochs"]):
    model.train()
    optimizer.zero_grad()
    output = model(train_x)

    loss_train = loss(output, train_y)
    acc_train = accuracy(output, train_y, binary)
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(validation_x)


    # calculate validation loss and accuracy.
    loss_val =  loss(output, validation_y)
    acc_val = accuracy(output, validation_y, binary)

    print(f"Epoch: {epoch}, Train Loss {round(loss_train.item(), 3)}, Train Acc: {round(acc_train.item(), 3)}, Valid Loss: {round(loss_val.item(), 3)}, Valid Acc: {round(acc_val.item(), 3)}")



args = {
    "optimizer": "adam",
    "dataset": "mnist",
    "criterion": "cross entropy",
    "epochs": 100
}

train_x, validation_x, test_x, train_y, validation_y, test_y = load_datasets(args)

model = MLNClassifier(32*32, 10)
train(model, args)

