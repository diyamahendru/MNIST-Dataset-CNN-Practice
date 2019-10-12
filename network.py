
import torch.nn as nn
import torch.nn.functional as f

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #The input channel is 1 and the output channel are 10.
        self.convolution1 =  nn.Conv2d(1, 10, 3) 
        #The input channel are 10 and the output channel are 20.
        self.convolution2 = nn.Conv2d(10, 20, 3) 
        self.maxpooling = nn.MaxPool2d(2)
        self.fullyConnected1 = nn.Linear(500, 50)
        self.fullyConnected2 = nn.Linear(50, 10)
        
    def forward(self, x):
        inSize = x.size(0)
        #The first convolution block
        x = f.relu(self.maxpooling(self.convolution1(x)))
        #The second conviolution block
        x = f.relu(self.maxpooling(self.convolution2(x)))
        #Flattening the tensor
        x = x.view(inSize, -1)
        #Fully connected layers
        x = self.fullyConnected1(x)
        x = self.fullyConnected2(x)
        return f.log_softmax(x)
    

        
