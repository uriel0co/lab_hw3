import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.transforms import NormalizeFeatures
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch_geometric.transforms as T
import pickle


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

def visualize(h, color):
    h = h.cpu()
    z = TSNE(n_components=2).fit_transform(h.numpy())

    print("finish_tsne")
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color[80000:].cpu(), cmap="Set2")
    plt.show()
    



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the dimensions of input and output
input_dim = 128  # Dimensionality of input features (x)
hidden_dim = 1024  # Dimensionality of hidden layer
output_dim = 40   # Number of classes in classification task
num_epochs = 300

model = GNNClassifier(input_dim, hidden_dim, output_dim).to(device)

# loss function (criterion) and optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# learning rate scheduler
scheduler = StepLR(optimizer, step_size=50, gamma=0.2)

dataset = torch.load('data/hw3/processed/data.pt')

transform = NormalizeFeatures()

# Random Node Features Perturbation
# perturbation = T.RemoveDuplicatedEdges()
# dataset = perturbation(dataset)

x = dataset.x.to(device)
edge_index = dataset.edge_index.to(device)
y = dataset.y.to(device)
train_mask = dataset.train_mask.to(device)
val_mask = dataset.val_mask.to(device)

model.train()

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(x, edge_index)
    loss = criterion(output[train_mask], y[train_mask].squeeze())
    loss.backward()
    optimizer.step()
    #scheduler.step()
    # # Validation
    # model.eval()
    # with torch.no_grad():
    #     val_output = model(x, edge_index)[val_mask]
    #     val_loss = criterion(val_output, y[val_mask].squeeze())

    print(f"Epoch: {epoch}, Training Loss: {loss.item()}")

model.eval()
with torch.no_grad():
    test_output = model(x, edge_index)[val_mask]
    _, predicted_labels = test_output.max(dim=1)


predicted_labels = predicted_labels.cpu()
ground_truth_labels = y[val_mask].squeeze().cpu()

# accuracy
correct = (predicted_labels == ground_truth_labels).sum().item()
total = ground_truth_labels.size(0)
accuracy = correct / total * 100

print(f"Accuracy: {accuracy:.2f}%")

# Save the model to a file
# with open('model.pkl', 'wb') as file:
#     pickle.dump(model, file)
#visualize(test_output, color=y)
