import torch
from GR_PINN.pinn import StarNet, sample_collocation, calculate_total_loss

model = StarNet()
physics_data, ic_data = sample_collocation(4)
loss = calculate_total_loss(model, physics_data, ic_data, M=0.5)
print(loss)
loss.backward()