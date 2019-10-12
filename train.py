
import torch.optim as opt
import torch.nn.functional as f
from torch.autograd import Variable
from network import Net
from dataloader import trainLoader, testLoader

learningRate = 0.01
momentum = 0.1  

model = Net()
optimizer = opt.SGD(model.parameters(), lr = learningRate, momentum = momentum)

def train(e):
    model.train()
    for batchIndex, (data, target) in enumerate(trainLoader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = f.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batchIndex % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                e, batchIndex * len(data), len(trainLoader.dataset),
                100. * batchIndex / len(trainLoader), loss.data[0]))
            
def test():
    model.eval()
    testLoss = 0
    c = 0
    for data, target in testLoader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        testLoss += f.nll_loss(output, target, size_average=False).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        c += pred.eq(target.data.view_as(pred)).cpu().sum()

    testLoss /= len(testLoader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        testLoss, c, len(testLoader.dataset),
        100. * c / len(testLoader.dataset)))


for e in range(1, 8):
    train(e)
    test()
