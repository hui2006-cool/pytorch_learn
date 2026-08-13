import torch
import torch.nn as nn

inputs = torch.tensor([1,2,3], dtype=torch.float32)
targets = torch.tensor([1,2,3], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss1 = nn.L1Loss()
loss2 = nn.MSELoss()

output1 = loss1(inputs, targets)
output2 = loss2(inputs, targets)

print(output1)
print(output2)