import pandas as pd
import numpy as np
import cdt
import networkx as nx
import os
from notears.notears import linear
from notears.notears import utils

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz

import matplotlib.pyplot as plt

def convert_cpdag_format(causal_graph):
    """
    Convert the CPDAG format from CausalGraph (cg.G.graph) to an adjacency matrix format.
    
    Args:
        causal_graph: A CausalGraph object with cg.G.graph as the adjacency matrix.
    
    Returns:
        A NumPy adjacency matrix where:
        - 1 represents an edge (directed or undirected).
        - 0 represents no edge.
    """
    # Extract the graph matrix
    graph_matrix = causal_graph.G.graph
    n = graph_matrix.shape[0]
    
    # Initialize the adjacency matrix
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(n):
            if graph_matrix[j, i] == 1 and graph_matrix[i, j] == -1:
                # Directed edge i -> j
                adjacency_matrix[i, j] = 1
            elif graph_matrix[i, j] == -1 and graph_matrix[j, i] == -1:
                # Undirected edge i -- j
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
            elif graph_matrix[i, j] == 1 and graph_matrix[j, i] == 1:
                # Bidirectional edge i <-> j
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
    
    return adjacency_matrix

data_dir = '/cim/ehoney/IFT6168_project/dcdi/data/perfect'

# List of folder names in the data directory
data_folder_names = [
    d
    for d in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, d))
    and d != 'sachs_intervention' and not d.endswith('.zip')
]

data_folder_names.sort()

out_dir = '/cim/ehoney/IFT6168_project/experiments/baseline'

lambda1 = 0.5
loss_type = 'l2'

metrics = {}

# Iterate through each data directory
for data_folder_name in data_folder_names:
    data_folder_path = os.path.join(data_dir, data_folder_name)
    print(f"Processing data directory: {data_folder_path}")

    output_dir = os.path.join(out_dir, data_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    metrics[data_folder_name] = {}
    
    # Iterate through the 10 datasets in the current directory
    for i in range(1, 11):
        metrics[data_folder_name][i] = {}

        # Load the DAG, CPDAG, and observational data
        dag_path = os.path.join(data_folder_path, f'DAG{i}.npy')
        cpdag_path = os.path.join(data_folder_path, f'CPDAG{i}.npy')
        obs_data_path = os.path.join(data_folder_path, f'data{i}.npy')
        
        dag = np.load(dag_path)
        cpdag = np.load(cpdag_path)
        obs_data = np.load(obs_data_path)

        ## PC algorithm
        
        # Estimate the CPDAG using the PC algorithm
        cg = pc(obs_data)
        
        # Convert the estimated CPDAG to adjacency matrix format
        estimated_cpdag = convert_cpdag_format(cg)

        # Save the estimated CPDAG
        output_path = os.path.join(output_dir, f'PC_alg_CPDAG{i}.npy')
        np.save(output_path, estimated_cpdag)

        pc_shd = cdt.metrics.SHD(estimated_cpdag, cpdag, double_for_anticausal=False)
        metrics[data_folder_name][i]['PC_SHD'] = pc_shd
        

        ## NOTEARS
        W_est = linear.notears_linear(obs_data, lambda1=lambda1, loss_type=loss_type)

        # Save the estimated W matrix
        output_path = os.path.join(output_dir, f'NOTEARS_W{i}.npy')
        np.save(output_path, W_est)

        estimated_dag = (W_est != 0).astype(int)

        notears_shd = cdt.metrics.SHD(estimated_dag, dag, double_for_anticausal=False)
        metrics[data_folder_name][i]['NOTEARS_SHD'] = notears_shd

        # Print SHDs for the current dataset
        print(f"Dataset {i}: PC SHD = {pc_shd}, NOTEARS SHD = {notears_shd}")

    # Compute the mean and variance of SHD values for the current directory
    pc_shd_values = [metrics[data_folder_name][i]['PC_SHD'] for i in range(1, 11)]
    notears_shd_values = [metrics[data_folder_name][i]['NOTEARS_SHD'] for i in range(1, 11)]
    mean_pc_shd = np.mean(pc_shd_values)
    variance_pc_shd = np.var(pc_shd_values)
    mean_notears_shd = np.mean(notears_shd_values)
    variance_notears_shd = np.var(notears_shd_values)

    # Display mean and variance of SHD values
    print(f"Mean PC SHD for {data_folder_name}: {mean_pc_shd}")
    print(f"Variance PC SHD for {data_folder_name}: {variance_pc_shd}")
    print(f"Mean NOTEARS SHD for {data_folder_name}: {mean_notears_shd}")
    print(f"Variance NOTEARS SHD for {data_folder_name}: {variance_notears_shd}")

# Save metrics as npy file inside output directory
metrics_output_path = os.path.join(out_dir, 'metrics.npy')
np.save(metrics_output_path, metrics)