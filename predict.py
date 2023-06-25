from dataset import *
import pickle
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


dataset = HW3Dataset(root='data/hw3/')
dataset = dataset[0]

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNClassifier, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(input_dim, hidden_dim).to(device)
        self.conv2 = GCNConv(hidden_dim, output_dim).to(device)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the dimensions of input and output
input_dim = 128  # Dimensionality of input features (x)
hidden_dim = 1024  # Dimensionality of hidden layer
output_dim = 40   # Number of classes in classification task
num_epochs = 300

x = dataset.x.to(device)
edge_index = dataset.edge_index.to(device)

model = GNNClassifier(input_dim, hidden_dim, output_dim).to(device)

# Import trained model
# Load the model from a file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

    
#classify to 1 of 40 labels
# Predict the data
model.eval()
with torch.no_grad():
    test_output = model(x, edge_index)
    _, predicted_labels = test_output.max(dim=1)


# Create a CSV file to store the predictions
with open("prediction.csv", "w") as f:
    f.write("idx,prediction\n")
    for i in range(len(predicted_labels)):
        f.write(f"{int(i)},{int(predicted_labels[i])}\n")
        
# Save prediction.csv file
