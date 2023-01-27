import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from src.paper import get_matrix_saliency, update_model_and_saliency_matrix


# Définir l'architecture du modèle
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 50)
        self.fc2 = nn.Linear(50, 5)
        self.fc3 = nn.Linear(5, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train(model, device, X_train_data, y_train_data, optimizer, epoch):
    criterion = nn.CrossEntropyLoss()
    model.train()

    # TODO:
    # X_train_data.to(device)
    # y_train_data.to(device)

    # Propager les données d'entraînement à travers le modèle
    y_pred = model(X_train_data)

    # fonction de coût
    loss = criterion(y_pred, y_train_data)

    # Réinitialiser les gradients et effectuer une étape d'optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test(model, X_test_data, y_test_data):
    # model.eval()
    criterion = nn.CrossEntropyLoss()
    # with torch.no_grad():
    # y_pred = model(X_test_data)
    # loss = criterion(y_pred, y_test_data)



    # Validation
    y_val_pred = model(X_test_data)
    val_loss = criterion(y_val_pred, y_test_data)

    accuracy = (y_val_pred.argmax(1) == y_test_data).float().mean()

    return val_loss.item(), accuracy.item(), val_loss.item()

    # Stocker les métriques d'entraînement pour cette époque
    # train_losses.append(loss.item())
    # train_accuracies.append(accuracy.item())
    # validation.append(val_loss.item())


def main():
    # Device setup
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Datasets
    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).long()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).long()

    # TODO
    # model = Net().to(device)
    model = Net()

    # Définir l'optimiseur et la fonction de coût
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Initialiser les tableaux pour stocker les métriques d'entraînement
    train_losses = []
    train_accuracies = []
    validation = []

    # Entraîner le modèle
    time0 = time.time()
    for epoch in range(1000):
        train(model=model,
              X_train_data=X_train_tensor,
              device=device,
              y_train_data=y_train_tensor,
              optimizer=optimizer,
              epoch=epoch
              )
        train_loss, train_accuracy, vali = test(model, X_test_data=X_test_tensor, y_test_data=y_test_tensor)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation.append(vali)

    weights = model.fc1.weight
    bias = model.fc1.bias
    coefs = model.fc2.weight[0]

    # Premier calcul de la matrice de saliency
    matrix_saliency = get_matrix_saliency(weights, bias, coefs)

    # Le prunning commence. Il suffit d'appeler autant de fois cette fonction que l'on veut :)
    for _ in range(10):
        matrix_saliency = update_model_and_saliency_matrix(model, matrix_saliency)

    y_pred = model(X_train_tensor)
    accuracy = (y_pred.argmax(1) == y_train_tensor).float().mean()
    print(f"Accuracy: {accuracy}")

    time1 = time.time()

    show_graphes = True
    if show_graphes:
        # Afficher les courbes accuracy et de loss
        plt.figure()
        plt.plot(train_losses, label='Training Loss', color='blue')
        plt.plot(validation, label='Validation Loss', color='red')
        plt.title('Evolution de la validation et du training', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(train_accuracies)
        plt.xlabel('Epoque')
        plt.ylabel('Accuracy')
        plt.title('Evolution de l\'accuracy', color='orange')
        plt.show()

    print("Nous avons donc un temps de:", time1 - time0)


if __name__ == '__main__':
    main()
