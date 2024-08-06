import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary


# Define the model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4 * 10, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 36 * 10)

    def forward(self, x):
        x = x.view(-1, 4 * 10)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(-1, 6, 6, 10)

# Load the trained model (assume the model is saved as 'model')
model = Model()
model.load_state_dict(torch.load('model'))
summary(model, (4 * 10,))
model.eval()

# One-hot encode the input data
def one_hot_encode(data, num_classes=10):
    data = np.array(data)
    one_hot_encoded = np.eye(num_classes)[data]
    return one_hot_encoded

# Define a function to perform inference
def predict(input_data):
    with torch.no_grad():
        input_tensor = torch.tensor(one_hot_encode(input_data), dtype=torch.float32).flatten().unsqueeze(0)
        output_tensor = model(input_tensor)
        output_tensor = output_tensor.squeeze(0).argmax(dim=2)  # Decode one-hot back to original values
        return output_tensor.numpy()

# Example input
input_data = [[1, 4], [7, 9]]
predicted_output = predict(input_data)
print("Predicted Output:")
print(predicted_output)