import torch
from torch.nn import Linear
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(data.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, 40)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

data = torch.load('data/hw3/processed/data.pt')


train_mask = data.train_mask
val_mask = data.val_mask

model = MLP(hidden_channels=16)
criterion = torch.nn.NLLLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask].squeeze())# Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      test_output = model(data.x)[val_mask]
      _, predicted_labels = test_output.max(dim=1)
      #pred = out.argmax(dim=1)  # Use the class with highest probability.
      #test_correct = pred[data.val_mask] == data.y[data.val_mask]
      predicted_labels = predicted_labels.cpu()
      ground_truth_labels = data.y[val_mask].squeeze().cpu()
      
      # Calculate the accuracy
      correct = (predicted_labels == ground_truth_labels).sum().item()
      total = ground_truth_labels.size(0)
      accuracy = correct / total * 100

      print(f"Accuracy: {accuracy:.2f}%")


for epoch in range(1, 51):
    loss = train()
    if epoch%10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    
test_acc = test()
