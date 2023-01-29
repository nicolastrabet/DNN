import torch


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


def pruning(model, nb, name1, name2):
    layer_obj_1 = getattr(model, name1)
    layer_obj_2 = getattr(model, name2)

    weights = layer_obj_1.weight
    bias = layer_obj_1.bias
    coefs = layer_obj_2.weight[0]

    # Premier calcul de la matrice de saliency
    matrix_saliency = get_matrix_saliency(weights, bias, coefs)

    # Le prunning commence. Il suffit d'appeler autant de fois cette fonction que l'on veut :)
    for _ in range(nb):
        matrix_saliency = update_model_and_saliency_matrix_with_param(model, matrix_saliency, name1, name2)
