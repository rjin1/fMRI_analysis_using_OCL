import torch
import torch.nn as nn

device = torch.device('cuda:0')

# a = torch.tensor([2], dtype=torch.float32).to(device)
# b = torch.tensor([2], dtype=torch.float32).to(device)

# c = a * b
model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(True), nn.Linear(4, 6), nn.ReLU(True))
model.to(device)

input = torch.randn([5, 2]).to(device)

a = model(input)

