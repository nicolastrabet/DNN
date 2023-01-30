from __future__ import print_function

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

from src.paper import pruning
from src.utils import human_bytes


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=50 * 7 * 7, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 50 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accu = 100. * correct / len(test_loader.dataset)

    return accu, test_loss


def main():
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 2
    lr = 1.0

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = LeNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    acc, loss = test(model, device, test_loader)
    print(f"\nAccuracy before pruning: {acc} (loss {loss})")

    torch.save(model.state_dict(), './results/mnist/model_before_pruning.pth')
    torch.save(optimizer.state_dict(), './results/mnist/optimizer_before_pruning.pth')

    time0 = time.time()
    pruning(model, 400, "fc1", "fc2")
    time1 = time.time()

    print(f"Time for the pruning : {round(time1 - time0)} secondes")

    model.to(device)
    acc, loss = test(model, device, test_loader)
    print(f"Accuracy after pruning: {acc} (loss {loss})\n")

    # Save model after pruning
    torch.save(model.state_dict(), './results/mnist/model_after_pruning.pth')
    torch.save(optimizer.state_dict(), './results/mnist/optimizer_after_pruning.pth')

    size_before_pruning = os.path.getsize("./results/mnist/model_before_pruning.pth")
    size_after_pruning = os.path.getsize("./results/mnist/model_after_pruning.pth")

    print(f"Size before pruning = {human_bytes(size_before_pruning)}\n"
          f"Size after pruning = {human_bytes(size_after_pruning)}\n"
          f"Différence = {human_bytes(size_before_pruning - size_after_pruning)}")


if __name__ == '__main__':
    main()
