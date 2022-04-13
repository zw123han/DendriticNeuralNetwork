import math
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision

def load_datasets(args):
  if(args.dataset == "minist"):
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
    train_dl = DataLoader(mnist_train, batch_size=bs)


    validation_dl = DataLoader(mnist_validation, batch_size = 100)

    test_dl = DataLoader(mnist_test, batch_size = 100)

    dataiter = iter(train_dl)
    train_x, train_y = dataiter.next()
    train_x = nn.Upsample(size=(32, 32))(train_x)



    dataiter = iter(validation_dl)
    validation_x, validation_y = dataiter.next()
    validation_x = nn.Upsample(size=(32, 32))(validation_x)


    dataiter = iter(test_dl)
    test_x, test_y = dataiter.next()
    test_x = nn.Upsample(size=(32, 32))(test_x)


    return train_x, validation_x, test_x, train_y, validation_y, test_y


def train(model, args):
  # model is the model we are going to train.
  # args is a dictionary of all the hyperparameters.


  train_losses = []
  valid_losses = []
  valid_accs = []
  # Load optimizer, and loss.
  optimizer = args['optimizer']()
  loss = args['loss']()
  def accuracy(output, labels):

    # From programming assignment 4.
    predictions = output.max(1)[1].type_as(labels)
    correct = predictions.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

  train_x, validation_x, test_x, train_y, validation_y, test_y = load_datasets(args)

  for epoch in range(args["epochs"]):
    model.train()
    optimizer.zero_grad()
    output = model(train_x)

    loss_train = loss(output, train_y)
    acc_train = accuracy(output, train_y)
    loss_train.backward()
    optimizer.step()

    model.eval()
    output = model(validation_x)


    # calculate validation loss and accuracy.
    loss_val =  loss(output, validation_y)
    acc_val = accuracy(output, validation_y)

    print("***********************Epoch: {} ***************************".format(str(epoch)))

    print("*****************************Train Loss: {}, Train Accuracy: {}".format(str(loss_train), str(acc_train)))

    print("*****************************Validation Loss: {}, Validation Accuracy: {}".format(str(loss_val), str(acc_val)))