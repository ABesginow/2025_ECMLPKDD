import gpytorch
from LODEGP.LODEGP import LODEGP, list_standard_models
import torch


train_x = torch.linspace(0, 1, 100)
train_y = torch.linspace(0, 1, 100)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
LODEGP(train_x, train_y, likelihood, 3, ODE_name="Bipendulum", verbose=True)

print(list_standard_models())