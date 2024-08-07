import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from grokfast import gradfilter_ma, gradfilter_ema
import numpy as np
import random
from torchsummary import summary


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the dataset
def generate_example():
    input_grid = [[random.randint(0, 9) for _ in range(2)] for _ in range(2)]
    output_grid = []
    for i in range(6):
        row = []
        if i % 2 == 0:
            row.extend(input_grid[0] * 3)
        else:
            row.extend(input_grid[1] * 3)
        output_grid.append(row)
    for i in range(2, 4):
        output_grid[i] = output_grid[i][::-1]
    return {"input": input_grid, "output": output_grid}
data = {"train": [generate_example() for _ in range(100)]}

def one_hot_encode(data, num_classes=10):
    data = np.array(data)
    one_hot_encoded = np.eye(num_classes)[data]
    return one_hot_encoded

class PuzzleDataset(Dataset):
    def __init__(self, data):
        self.data = data["train"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        input_tensor = torch.tensor(one_hot_encode(sample["input"]), dtype=torch.float32).flatten()
        output_tensor = torch.tensor(one_hot_encode(sample["output"]), dtype=torch.float32)
        return input_tensor, output_tensor

# Split the data into training and validation sets
train_data = data["train"][:int(len(data["train"])*.9)]
val_data = data["train"][int(len(data["train"])*.1):]

data_train = {"train": train_data}
data_val = {"train": val_data}

# Create the DataLoaders
train_dataset = PuzzleDataset(data_train)
val_dataset = PuzzleDataset(data_val)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Define the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4 * 10, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 36 * 10)

    def forward(self, x):
        x = x.view(-1, 4 * 10)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 6, 6, 10)

model = Model().to(device)
summary(model, (4 * 10,), device="cuda")

# Hyperparameters
alpha = 0.98
lamb = 80.0
num_epochs = 300
lr = 0.000007
weight_decay = 0.0005

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 3)
            _, targets_max = torch.max(targets, 3)
            total += targets.nelement() / 10
            correct += (predicted == targets_max).sum().item()
    return correct / total

grads = None
train_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        grads = gradfilter_ema(model, grads=grads, alpha=alpha, lamb=lamb)

        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # Calculate accuracies
    train_accuracy = calculate_accuracy(model, train_loader)
    val_accuracy = calculate_accuracy(model, val_loader)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), "model")

# Plot the metrics
fig, ax1 = plt.subplots(figsize=(10, 7))

# Plot Training Loss
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Training Loss", color="tab:cyan")
ax1.plot(range(1, num_epochs + 1), train_losses, label="Training Loss", color="tab:cyan")
ax1.tick_params(axis="y", labelcolor="tab:cyan")

# Create a second y-axis for accuracy
ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy")
ax2.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy", color="tab:blue")
ax2.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy", color="tab:orange")
ax2.tick_params(axis="y")

# Add titles and legends
fig.tight_layout()
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.title("Loss & Accuracy")

plt.show()