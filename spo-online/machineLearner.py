import torch
from torch.autograd import Variable

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class MLP(torch.nn.Module):
  '''
    Multilayer Perceptron for regression.
  '''
  def __init__(self):
    super().__init__()
    self.layers = torch.nn.Sequential(
      torch.nn.Linear(5, 10),
      torch.nn.ReLU(),
      torch.nn.Linear(10, 5),
      torch.nn.ReLU(),
      torch.nn.Linear(5, 1)
    )


  def forward(self, x):
    '''
      Forward pass
    '''
    return self.layers(x)