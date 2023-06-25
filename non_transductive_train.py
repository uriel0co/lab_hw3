import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNClassifier, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Define the device to be used (CPU or GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the dimensions of your input and output
input_dim = 128  # Dimensionality of input features (x)
hidden_dim = 64  # Dimensionality of hidden layer
output_dim = 40   # Number of classes in your classification task
num_epochs = 50

# Create the model
model = GNNClassifier(input_dim, hidden_dim, output_dim)

# Define the loss function (criterion) and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

dataset = torch.load('data/hw3/processed/data.pt')

# Assuming you have the data tensors in the following variables
x = dataset.x
edge_index = dataset.edge_index
y = dataset.y
train_mask = dataset.train_mask
val_mask = dataset.val_mask

# Set the model in training mode
model.train()

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x, edge_index)[train_mask]
    loss = criterion(output, y[train_mask].squeeze())
    loss.backward()
    optimizer.step()

    # # Validation
    # model.eval()
    # with torch.no_grad():
    #     val_output = model(x, edge_index)[val_mask]
    #     val_loss = criterion(val_output, y[val_mask].squeeze())

    #print(f"Epoch: {epoch}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

# Assuming you have trained your model and obtained the test predictions
model.eval()
with torch.no_grad():
    test_output = model(x, edge_index)[val_mask]
    _, predicted_labels = test_output.max(dim=1)

# Convert the predicted labels and ground truth labels to CPU tensors if necessary
predicted_labels = predicted_labels.cpu()
ground_truth_labels = y[val_mask].squeeze().cpu()

# Calculate the accuracy
correct = (predicted_labels == ground_truth_labels).sum().item()
total = ground_truth_labels.size(0)
accuracy = correct / total * 100

print(f"Accuracy: {accuracy:.2f}%")
