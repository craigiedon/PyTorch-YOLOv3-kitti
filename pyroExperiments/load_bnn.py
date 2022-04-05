import pyro
from pyro.infer import Predictive
import torch
import numpy as np
from pyro.infer.autoguide import AutoDiagonalNormal
import matplotlib.pyplot as plt

from pyroExperiments.regression_test import Model, trans_func

model = Model(20, 20)
print("named params", list(model.named_parameters()))
guide = AutoDiagonalNormal(model)

p_store = pyro.get_param_store()
guide = torch.load("../saved_models/guide.pt")
p_store.load("../saved_models/guide_params.save")

x_data = torch.tensor(np.linspace(0, 2 * np.pi, 1000), dtype=torch.float)
x_mean = x_data.mean()
x_std = x_data.std()
x_data = (x_data - x_mean) / x_std
print(x_mean, x_std)
fx_data = trans_func(x_data)
y_data = fx_data + 0.2 * torch.cat((torch.randn(len(x_data) // 2), torch.zeros(len(x_data) // 2)))

predictive = Predictive(model, guide=guide, num_samples=1000)
x_pred_points = (torch.linspace(-1.5 * np.pi, 3.5 * np.pi, 1000) - x_mean) / x_std
preds = predictive(x_pred_points.unsqueeze(1))
# pred_summary = summary(samples)

y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
y_std = preds['obs'].T.detach().numpy().std(axis=1)

plt.scatter(x_data.detach().numpy(), y_data.detach().numpy(), marker='x', color='green', alpha=0.2)
plt.plot(x_pred_points.detach().numpy(), trans_func(x_pred_points).detach().numpy(), color='orange')
plt.plot(x_pred_points.detach().numpy(), y_pred, color='b')
plt.fill_between(x_pred_points, y_pred - y_std, y_pred + y_std, alpha=0.5, color='b')
plt.ylim(-3, 3)
plt.show()
