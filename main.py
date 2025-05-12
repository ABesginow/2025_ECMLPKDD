import gpytorch
from LODEGP.LODEGP import LODEGP, list_standard_models
from sage.all import *
import sage
import torch


train_x = torch.linspace(0, 1, 100)
train_y = torch.linspace(0, 1, 100)
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)

print(list_standard_models())












R = QQ['x']; (x,) = R._first_ngens(1)
# Linearized bipendulum
#A = matrix(R, Integer(2), Integer(3), [x**2 + 9.81, 0, -1, 0, x**2+4.905, -1/2])
#A = matrix(R, Integer(1), Integer(3), [x**2 + 9.81, 0, -1])
#A = matrix(R, Integer(1), Integer(3), [0, x**2+4.905, -1/2])
#A = matrix(R, Integer(1), Integer(3), [0, 0, 0])
#LODEGP(train_x, train_y, likelihood, 3, A=A, verbose=True)

LODEGP(train_x, train_y, likelihood, 3, ODE_name="Bipendulum Parameterized", verbose=True)
#LODEGP(train_x, train_y, likelihood, 3, ODE_name="Heating", verbose=True)



