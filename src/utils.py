import math as m
import torch
import torch.nn.functional as F

MULTIPLES = ["B", "k{}B", "M{}B", "G{}B", "T{}B", "P{}B", "E{}B", "Z{}B", "Y{}B"]


def human_bytes(i, binary=False, precision=2):
    base = 1024 if binary else 1000
    multiple = m.trunc(m.log2(i) / m.log2(base))
    value = i / m.pow(base, multiple)
    suffix = MULTIPLES[multiple].format("i" if binary else "")
    return f"{value:.{precision}f} {suffix}"

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