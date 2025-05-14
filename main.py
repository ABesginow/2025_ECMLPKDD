import gpytorch
from LODEGP.LODEGP import LODEGP
from helpers.training_functions import granso_optimization
from helpers.plotting_functions import plot_single_input_gp_posterior
from sage.all import *
import sage
import torch


# Generate a solution to the physical system
START = 2
END = 12
COUNT = 100
train_x = torch.linspace(START, END, COUNT)

y0_func = lambda x: float(781/8000)*torch.sin(x)/x - float(1/20)*torch.cos(x)/x**2 + float(1/20)*torch.sin(x)/x**3
y1_func = lambda x: float(881/8000)*torch.sin(x)/x - float(1/40)*torch.cos(x)/x**2 + float(1/40)*torch.sin(x)/x**3
y2_func = lambda x: float(688061/800000)*torch.sin(x)/x - float(2543/4000)*torch.cos(x)/x**2 + float(1743/4000)*torch.sin(x)/x**3 - float(3/5)*torch.cos(x)/x**4 + float(3/5)*torch.sin(x)/x**5 
y0 = y0_func(train_x)
y1 = y1_func(train_x)
y2 = y2_func(train_x)
train_y = torch.stack([y0, y1, y2], dim=-1)

# Bipendulum versions to test:
# "Bipendulum", "Bipendulum first equation", "Bipendulum second equation", "Bipendulum Parameterized", "No system"
for system_name in ["Bipendulum", "Bipendulum first equation", "Bipendulum second equation", "Bipendulum Parameterized", "No system"]:
    print(f"Testing system: {system_name}")
    # 1. Model definition
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name=system_name, verbose=True, system_parameters={"l1": 1.0, "l2": 2.0})

    likelihood_MAP = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
    model_MAP = LODEGP(train_x, train_y, likelihood_MAP, 3, ODE_name=system_name, verbose=True, system_parameters={"l1": 1.0, "l2": 2.0})


    # 2. Model training (10 random restarts with PyGRANSO) (try MLL and MAP with uninformed prior)
    model.train()
    likelihood.train()
    unscaled_neg_MLL, model, likelihood, training_logs = granso_optimization(model, likelihood, train_x, train_y, random_restarts=0, uninformed=True, logarithmic_reinit=True, verbose=True, MAP=False)

    # print model parameters
    print("Model parameters after training:")
    print(list(model.named_parameters()))

    model_MAP.train()
    likelihood_MAP.train()
    #unscaled_neg_MLL, model_MAP, likelihood_MAP, training_logs = granso_optimization(model_MAP, likelihood_MAP, train_x, train_y, random_restarts=10, uninformed=True, logarithmic_reinit=True, verbose=True, MAP=True)

    print("Model parameters after training:")
    print(list(model_MAP.named_parameters()))

    # 3. Draw model posterior
    test_x = torch.linspace(START, END, COUNT)
    plot_colors = ['red', 'green', 'blue']
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        import pdb; pdb.set_trace()
        observed_pred = likelihood(model(test_x))
        pred_mean = observed_pred.mean
        pred_var = observed_pred.covariance_matrix
        pred_var = torch.diagonal(pred_var)
        # Reslice the pred_var to have a tensor of the same shape as pred_mean
        # Do this by putting every n-th element in the n-th dimension
        pred_var = pred_var.view(pred_mean.shape[0], -1)
    
    MLL_model_posterior_fig = plot_single_input_gp_posterior(train_x, train_y, test_x, pred_mean, pred_var, n_std=2, show=False, return_fig=True, fig=None, ax=None, colors=plot_colors, ncols=3, figsize=(20, 5), titles=["$f_1$", "$f_2$", "$u$"], xlabel="Time", ylabel="Output")

    model_MAP.eval()
    likelihood_MAP.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood_MAP(model_MAP(test_x))
        pred_mean = observed_pred.mean
        pred_var = observed_pred.covariance_matrix
        pred_var = torch.diagonal(pred_var)
        # Reslice the pred_var to have a tensor of the same shape as pred_mean
        # Do this by putting every n-th element in the n-th dimension
        pred_var = pred_var.view(pred_mean.shape[0], -1)
    
    MAP_model_posterior_fig = plot_single_input_gp_posterior(train_x, train_y, test_x, pred_mean, pred_var, n_std=2, show=False, return_fig=True, fig=None, ax=None, colors=plot_colors, ncols=3, figsize=(20, 5), titles=["$f_1$", "$f_2$", "$u$"], xlabel="Time", ylabel="Output")

    # Store the figures
    MLL_model_posterior_fig.savefig(f"MLL_model_posterior_{system_name}.png")
    MAP_model_posterior_fig.savefig(f"MAP_model_posterior_{system_name}.png")

    # 4. Calculate precision? i.e. finite difference of samples on eval points