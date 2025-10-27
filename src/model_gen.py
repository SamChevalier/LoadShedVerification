import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd

# --- Hyperparams --------------------------------------------------------

epochs = 150

hidden_dim_depth = 2048

batch_sizes = 1

dropout_percent = 0.2

learn_rate = 1e-4

output_filename = "data_file_24bus.h5"
normalization_filename = "24bus_normalization_values.h5"

percent_train = 0.8
percent_val = 0.1
percent_test = 0.1

# ----------------------------------------------------------------------

# Import and format data
if (percent_test + percent_train + percent_val != 1):
    exit(code = "Data split unevenly!")

x_sample_temp = []
y_sample_temp = []
with h5py.File(output_filename, "r") as f:
    for i in f["sample_data"].keys():
        if i != "num_samples" and i != "index" and len(f["sample_data"][i]["branch"]) == 2:
            temp_power_risk = torch.tensor(np.array(f["sample_data"][i]["branch"]["power_risk"][()]), dtype=torch.float32)
            temp_qd = torch.tensor(np.array(f["sample_data"][i]["load"]["qd"][()]), dtype=torch.float32)
            temp_pd = torch.tensor(np.array(f["sample_data"][i]["load"]["pd"][()]), dtype=torch.float32)
            temp_alpha = torch.tensor(np.array(f["sample_data"][i]["alpha"][()]), dtype=torch.float32)
            x = torch.cat([temp_power_risk, temp_qd, temp_pd, temp_alpha], dim = 0)
            y = torch.tensor(np.array(f["sample_data"][i]["branch"]["status"][()]), dtype=torch.float32)
            x_sample_temp.append(x)
            y_sample_temp.append(y)

class OpsDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, truths):
        self.inputs = inputs
        self.truths = truths
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.truths[idx]

num_branches = len(y_sample_temp[1])
input_dim = len(x_sample_temp[1])

# Try normalizing inputs 
mean = torch.mean(torch.stack(x_sample_temp), dim=0)
std = torch.std(torch.stack(x_sample_temp), dim=0) + 1e-6
x_sample_temp = [(x - mean) / std for x in x_sample_temp]

with h5py.File(normalization_filename, "w") as f:
    mean.cpu()
    std.cpu()
    f["mean"] = mean.cpu().numpy()
    f["std"] = std.cpu().numpy()

print(f"# of datapoints = {len(x_sample_temp)}")

data = OpsDataset(x_sample_temp, y_sample_temp)

train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(data, (percent_train, percent_test, percent_val))

model = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim_depth),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_percent),
                            torch.nn.Linear(hidden_dim_depth, hidden_dim_depth),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_percent),
                            torch.nn.Linear(hidden_dim_depth, hidden_dim_depth),
                            torch.nn.ReLU(),
                            torch.nn.Dropout(dropout_percent),
                            torch.nn.Linear(hidden_dim_depth, num_branches))
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_sizes, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_sizes, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_sizes, drop_last=True)

train_losses = []
val_losses = []

criterion = nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for i in range(epochs):
    model.train()
    train_loss = 0.0
    for input, truth in train_dataloader:
        if torch.cuda.is_available():
            input, truth = input.to(device), truth.to(device)
            model.cuda()
        else:
            model.cpu()

        optimizer.zero_grad()
    
        outputs = model(input)
        loss = criterion(outputs, truth)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_losses.append(train_loss / len(train_dataloader))  

    model.eval()
    val_loss = 0.0
    for input, truth in val_dataloader:
        if torch.cuda.is_available():
            input, truth = input.to(device), truth.to(device)
        
        output = model(input)
        loss = criterion(outputs, truth)

        val_loss += loss.item()
    val_losses.append(val_loss / len(val_dataloader)) 
    print(f"[Epoch {i+1}] Training loss: {train_loss / len(train_dataloader):.2f}, Validation loss: {val_loss / len(val_dataloader):.2f}")

preds_vec = []
truths_vec = []
with torch.no_grad():
    for input, truth in val_dataloader:
        input, truth = input.to(device), truth.to(device)
        outputs = model(input)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        for i in preds:
            preds_vec.append(i)
        for i in truth:
            truths_vec.append(i)

total = len(preds_vec)
correct = 0

for e, i in enumerate(preds_vec):
    if (i == truths_vec[e]).sum().item() == num_branches:
        correct += 1

print(f"Total Accuracy Average: {correct / total * 100:.4f}")

total = 0
correct = 0
with torch.no_grad():
    for input, truth in val_dataloader:
        input, truth = input.to(device), truth.to(device)
        outputs = model(input)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == truth).sum().item()
        total += torch.numel(truth)
print(f"Branchwise Accuracy Average: {correct / total * 100:.4f}")
        

plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("118_bus_2048node.png", dpi=300)
#plt.show()

model.eval()

new_model = nn.Sequential(model[0], model[1], model[3], model[4], model[6], model[7], model[9])

torch.save(new_model, "118_bus_2048node.pt")


