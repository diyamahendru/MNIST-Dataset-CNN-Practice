
import torch
import torchvision.datasets as d
import torchvision.transforms as t

batchSize = 64

trainLoader = torch.utils.data.DataLoader(d.MNIST(root = "./data", 
                                                   train = True, 
                                                   transform = t.ToTensor(), 
                                                   download = True), 
                                            batch_size = batchSize,
                                            shuffle = True)

testLoader = torch.utils.data.DataLoader(d.MNIST(root = "./data", 
                                                  train= False,
                                                  transform = t.ToTensor(),
                                                  download = True),
                                            batch_size = batchSize,
                                            shuffle = True)

