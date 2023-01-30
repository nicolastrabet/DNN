import os
import time
import torch
from matplotlib import pyplot as plt

from src.utils import test


def calculate_saliency(weight_i, bias_i, weight_j, bias_j, coef_j):
    epsilon = weight_i - weight_j + bias_i - bias_j
    saliency = (coef_j ** 2) * torch.norm(epsilon, 2) ** 2
    return saliency


def get_matrix_saliency(weights, bias, coefs):
    # Calculate the saliency matrix for the given layer
    saliency_matrix = torch.empty((weights.shape[0], weights.shape[0]))
    for i in range(weights.shape[0]):
        for j in range(weights.shape[0]):
            saliency_matrix[i][j] = calculate_saliency(weights[i], bias[i], weights[j], bias[j], coefs[j])
            # calculate_saliency(weights[i], bias[i], weights[j], bias[j], coefs[j])
    return saliency_matrix


def get_smallest_saliency_id(saliency_matrix):
    lowest_saliency = float("inf")
    lowest_saliency_indices = (0, 0)

    for i in range(saliency_matrix.shape[0]):
        for j in range(saliency_matrix.shape[0]):
            # On ne veut pas la diagonale
            if j != i:
                # Calculate the saliency for the current pair of weight sets
                current_saliency = saliency_matrix[i][j]
                # Update the lowest saliency and the corresponding indices if necessary
                if current_saliency < lowest_saliency:
                    lowest_saliency = current_saliency
                    lowest_saliency_indices = (i, j)
    return lowest_saliency_indices


def update_model_and_saliency_matrix(model, saliency_matrix):
    id = get_smallest_saliency_id(saliency_matrix)
    i, j = id[0], id[1]

    weights = model.fc1.weight
    bias = model.fc1.bias
    coefs = model.fc2.weight[0]

    model.fc1.weight.data = torch.cat((weights[:j], weights[j + 1:]))
    model.fc1.bias.data = torch.cat((bias[:j], bias[j + 1:]))

    for nb in range(model.fc2.weight.shape[0]):
        if nb == 0:
            model.fc2.weight.data[nb][i] += model.fc2.weight.data[nb][j]
            new_fc2 = torch.cat((model.fc2.weight.data[nb][:j], model.fc2.weight.data[nb][j + 1:]))
        else:
            model.fc2.weight.data[nb][i] += model.fc2.weight.data[nb][j]
            new_fc2 = torch.cat(
                (new_fc2, torch.cat((model.fc2.weight.data[nb][:j], model.fc2.weight.data[nb][j + 1:]))))

    model.fc2.weight.data = torch.reshape(new_fc2, [10, int(int(len(new_fc2)) / 10)])

    # Update the saliency matrix by removing the j-th column and row
    saliency_matrix = torch.cat((saliency_matrix[:i], saliency_matrix[i + 1:]))
    saliency_matrix = torch.cat((saliency_matrix[:, :j], saliency_matrix[:, j + 1:]), dim=1)

    weights = model.fc1.weight
    bias = model.fc1.bias
    coefs = model.fc2.weight[0]

    # Update the saliency values for the remaining weight pairs
    for k in range(saliency_matrix.shape[0]):
        saliency_matrix[k][j] = calculate_saliency(weights[k], bias[k], weights[i], bias[i], coefs[i])

    return saliency_matrix


def update_model_and_saliency_matrix_with_param(model, saliency_matrix, name1, name2):
    id = get_smallest_saliency_id(saliency_matrix)
    i, j = id[0], id[1]

    layer_obj_1 = getattr(model, name1)
    layer_obj_2 = getattr(model, name2)
    degub = False
    if degub:
        print(f"Type: {layer_obj_1}")
        print(f"Type: {type(layer_obj_1)}")

        print(f"Type: {layer_obj_2}")
        print(f"Type: {type(layer_obj_2)}")

    weights = layer_obj_1.weight
    bias = layer_obj_1.bias
    coefs = layer_obj_2.weight[0]

    layer_obj_1.weight.data = torch.cat((weights[:j], weights[j + 1:]))
    layer_obj_1.bias.data = torch.cat((bias[:j], bias[j + 1:]))

    for nb in range(layer_obj_2.weight.shape[0]):
        if nb == 0:
            layer_obj_2.weight.data[nb][i] += layer_obj_2.weight.data[nb][j]
            new_fc2 = torch.cat((layer_obj_2.weight.data[nb][:j], layer_obj_2.weight.data[nb][j + 1:]))
        else:
            layer_obj_2.weight.data[nb][i] += layer_obj_2.weight.data[nb][j]
            new_fc2 = torch.cat(
                (new_fc2, torch.cat((layer_obj_2.weight.data[nb][:j], layer_obj_2.weight.data[nb][j + 1:]))))

    layer_obj_2.weight.data = torch.reshape(new_fc2, [10, int(int(len(new_fc2)) / 10)])

    # Update the saliency matrix by removing the j-th column and row
    saliency_matrix = torch.cat((saliency_matrix[:i], saliency_matrix[i + 1:]))
    saliency_matrix = torch.cat((saliency_matrix[:, :j], saliency_matrix[:, j + 1:]), dim=1)

    weights = layer_obj_1.weight
    bias = layer_obj_1.bias
    coefs = layer_obj_2.weight[0]

    # Update the saliency values for the remaining weight pairs
    for k in range(saliency_matrix.shape[0]):
        saliency_matrix[k][j] = calculate_saliency(weights[k], bias[k], weights[i], bias[i], coefs[i])

    return saliency_matrix


def get_saliency_smallest_id(weights, bias, coefs):
    # Initialize the lowest saliency and the corresponding indices
    lowest_saliency = float("inf")
    lowest_saliency_indices = (0, 0)

    for i in range(weights.shape[0]):
        for j in range(weights.shape[0]):
            # On ne veut pas la diagonale
            if j != i:
                # Calculate the saliency for the current pair of weight sets
                current_saliency = calculate_saliency(weights[i], bias[i], weights[j], bias[j], coefs[j])
                # Update the lowest saliency and the corresponding indices if necessary
                if current_saliency < lowest_saliency:
                    lowest_saliency = current_saliency
                    lowest_saliency_indices = (i, j)
    return lowest_saliency_indices

def plot_accuracy_vs_neurons_pruned(accuracy, neurons_pruned):
    plt.plot(neurons_pruned, accuracy)
    plt.xlabel('Number of Neurons Pruned')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Number of Neurons Pruned')
    plt.show()

def pruning(model, nb, name1, name2, device, optimizer, test_loader, plot=False):
    time0 = time.time()
    neurones_pruned = [i for i in range(50, nb + 1, 50)]
    accuracy = []
    pruning_durations = []
    Compression = []
    layer_obj_1 = getattr(model, name1)
    layer_obj_2 = getattr(model, name2)

    weights = layer_obj_1.weight
    bias = layer_obj_1.bias
    coefs = layer_obj_2.weight[0]

    # Premier calcul de la matrice de saliency
    matrix_saliency = get_matrix_saliency(weights, bias, coefs)

    # Le prunning commence. Il suffit d'appeler autant de fois cette fonction que l'on veut :)
    for i in range(nb + 1):
        matrix_saliency = update_model_and_saliency_matrix_with_param(model, matrix_saliency, name1, name2)
        if i in neurones_pruned and plot:
            time1 = time.time()
            model.to(device)
            acc, loss = test(model, device, test_loader)
            accuracy.append(acc)
            pruning_durations.append(time1 - time0)
            # Save model after pruning
            torch.save(model.state_dict(), './results/mnist/model_after_pruning.pth')
            torch.save(optimizer.state_dict(), './results/mnist/optimizer_after_pruning.pth')
            size_before_pruning = os.path.getsize("./results/mnist/model_before_pruning.pth")
            size_after_pruning = os.path.getsize("./results/mnist/model_after_pruning.pth")
            compression = size_before_pruning / size_after_pruning
            Compression.append(compression)
            print(f"Number of pruned neurons: {i}")
            print(f"Time for the pruning : {round(time1 - time0)} secondes")
            print(f"Accuracy after pruning: {acc} (loss {loss})\n")
            print(f"Compression : {compression}")

    print("coucou")
    return accuracy, neurones_pruned,pruning_durations, Compression
