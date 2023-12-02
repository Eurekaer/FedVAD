import torch
import numpy as np
from sklearn.cluster import KMeans
def Dist(a, b):
    return torch.norm(a - b, dim=1)

def cluster_assignment_update(params_matrix, bert_features, cluster_centers_w, cluster_centers_o, alpha, beta):
    num_clients = params_matrix.shape[0]
    num_clusters = cluster_centers_w.shape[0]
    assignment = torch.zeros(num_clients, num_clusters)

    for i in range(num_clients):
        model_weights = torch.tensor(params_matrix[i]).float()
        combined_distribution = torch.tensor(bert_features[i]).float()

        distances = alpha * Dist(model_weights, cluster_centers_w) + beta * Dist(combined_distribution, cluster_centers_o)
        cluster_index = torch.argmin(distances)
        assignment[i, cluster_index] = 1

    return assignment

def update_cluster_centers(r_ik, params_matrix, bert_features):
    r_sum = torch.sum(r_ik, dim=0, keepdim=True)  # Shape (1, num_clusters)
    r_sum += 1e-8 * (r_sum == 0).float()  # Prevent division by zero
    new_cluster_centers_w = torch.mm(r_ik.t(), torch.tensor(params_matrix).float()) / r_sum.t()
    new_cluster_centers_o = torch.mm(r_ik.t(), torch.tensor(bert_features).float()) / r_sum.t()
    return new_cluster_centers_w, new_cluster_centers_o
