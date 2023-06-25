import torch
from torch_geometric.data import Data

def create_data():
    # Load the graph dataset from the .pt file
    dataset = torch.load('data/hw3/processed/data.pt')

    # Access the graph data
    x = dataset.x  # Node features
    y = dataset.y  # Node labels (if available)
    edge_index = dataset.edge_index  # Graph connectivity

    # Optional: Access additional data such as edge features, etc.
    # edge_attr = dataset.edge_attr

    # Create a PyG Data object
    data = Data(x=x, y=y, edge_index=edge_index)

    # Optional: Set additional attributes of the Data object if available
    # data.edge_attr = edge_attr

    return data

