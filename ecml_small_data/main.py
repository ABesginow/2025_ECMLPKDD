import gpytorch
import itertools
from lodegp import LODEGP
from helpers.training_functions import granso_optimization
from helpers.plotting_functions import plot_single_input_gp_posterior
from helpers.util_functions import central_difference
from sage.all import *
import sage
import torch

torch.set_default_dtype(torch.float64)


for ((START, END), COUNT, noise_level) in itertools.product([(2, 12), (2, 3)], [1, 2, 5, 10, 20, 50, 100], [0.0, 0.1, 0.2, 0.3]):
    train_x = torch.linspace(START, END, COUNT)

    y0_func = lambda x: float(781/8000)*torch.sin(x)/x - float(1/20)*torch.cos(x)/x**2 + float(1/20)*torch.sin(x)/x**3
    y1_func = lambda x: float(881/8000)*torch.sin(x)/x - float(1/40)*torch.cos(x)/x**2 + float(1/40)*torch.sin(x)/x**3
    y2_func = lambda x: float(688061/800000)*torch.sin(x)/x - float(2543/4000)*torch.cos(x)/x**2 + float(1743/4000)*torch.sin(x)/x**3 - float(3/5)*torch.cos(x)/x**4 + float(3/5)*torch.sin(x)/x**5 
    y0 = y0_func(train_x) 
    y1 = y1_func(train_x)
    y2 = y2_func(train_x)
    y0 = y0 + torch.randn_like(train_x)*(torch.max(y0)*noise_level)
    y1 = y1 + torch.randn_like(train_x)*(torch.max(y1)*noise_level)
    y2 = y2 + torch.randn_like(train_x)*(torch.max(y2)*noise_level)
    train_y = torch.stack([y0, y1, y2], dim=-1)

    # Bipendulum versions to test:
    # "Bipendulum", "Bipendulum first equation", "Bipendulum second equation", "Bipendulum Parameterized", "No system"
    for (system_name, length_params) in itertools.chain(itertools.product(["Bipendulum", "Bipendulum first equation", "Bipendulum second equation", "Bipendulum Sum eq2 diffed", "Bipendulum moon gravitation"], [[1.0, 2.0], [1.0, 3.0], [2.0, 3.0]]), [("Bipendulum Parameterized", [1.0, 2.0]),  ("No system", [1.0, 2.0])]):
        l1_param_val = length_params[0]
        l2_param_val = length_params[1]
        print("======================================")
        print(f"Running new setting")
        print("======================================")
        print(f"{system_name}_l1-{l1_param_val}_l2-{l2_param_val}_{START}-{END}-{COUNT}_{noise_level}")
        print("======================================")
        print("======================================")
        
        # 1. Model definition
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
        model = LODEGP.LODEGP(train_x, train_y, likelihood, 3, ODE_name=system_name, verbose=False, system_parameters={"l1": l1_param_val, "l2": l2_param_val})

        likelihood_MAP = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
        model_MAP = LODEGP.LODEGP(train_x, train_y, likelihood_MAP, 3, ODE_name=system_name, verbose=False, system_parameters={"l1": l1_param_val, "l2": l2_param_val})


        # 2. Model training (10 random restarts with PyGRANSO) (try MLL and MAP with uninformed prior)
        model.train()
        likelihood.train()
        unscaled_neg_MLL, model, likelihood, training_logs = granso_optimization(model, likelihood, train_x, train_y, random_restarts=1, uninformed=True, logarithmic_reinit=True, verbose=False, MAP=False)

        # print model parameters
        print("Model parameters after training:")
        print(list(model.named_parameters()))

        model_MAP.train()
        likelihood_MAP.train()
        unscaled_neg_MAP, model_MAP, likelihood_MAP, training_logs = granso_optimization(model_MAP, likelihood_MAP, train_x, train_y, random_restarts=1, uninformed=True, logarithmic_reinit=True, verbose=False, MAP=True)

        print("Model parameters after training:")
        print(list(model_MAP.named_parameters()))

        # 3. Draw model posterior
        test_x = torch.linspace(0, 15, 100)
        plot_colors = ['red', 'green', 'blue']
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            pred_mean = observed_pred.mean
            pred_var = observed_pred.covariance_matrix
            pred_var = torch.diagonal(pred_var)
            # Reslice the pred_var to have a tensor of the same shape as pred_mean
            # Do this by putting every n-th element in the n-th dimension
            pred_var = pred_var.view(pred_mean.shape[0], -1)
        
        MLL_model_posterior_fig, MLL_model_posterior_ax = plot_single_input_gp_posterior(train_x, train_y, test_x, pred_mean, pred_var, n_std=2, show=False, return_fig=True, fig=None, ax=None, colors=plot_colors, ncols=3, figsize=(20, 5), titles=["$f_1$", "$f_2$", "$u$"], xlabel="Time", ylabel="Output")
        
        # Create and draw the ground truth as red dashed
        MLL_model_posterior_ax[0].plot(test_x, y0_func(test_x), color="red") 
        MLL_model_posterior_ax[1].plot(test_x, y1_func(test_x), color="red") 
        MLL_model_posterior_ax[2].plot(test_x, y2_func(test_x), color="red") 

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
        
        MAP_model_posterior_fig, MAP_model_posterior_ax = plot_single_input_gp_posterior(train_x, train_y, test_x, pred_mean, pred_var, n_std=2, show=False, return_fig=True, fig=None, ax=None, colors=plot_colors, ncols=3, figsize=(20, 5), titles=["$f_1$", "$f_2$", "$u$"], xlabel="Time", ylabel="Output")

        # Create and draw the ground truth as red dashed
        MAP_model_posterior_ax[0].plot(test_x, y0_func(test_x), color="red") 
        MAP_model_posterior_ax[1].plot(test_x, y1_func(test_x), color="red") 
        MAP_model_posterior_ax[2].plot(test_x, y2_func(test_x), color="red") 

        # Store the figures
        MLL_model_posterior_fig.savefig(f"results/figures/MLL_model_posterior_{system_name}_l1-{l1_param_val}_l2-{l2_param_val}_{START}-{END}-{COUNT}_{noise_level}.png")
        MAP_model_posterior_fig.savefig(f"results/figures/MAP_model_posterior_{system_name}_l1-{l1_param_val}_l2-{l2_param_val}_{START}-{END}-{COUNT}_{noise_level}.png")

        # 4. Calculate precision? i.e. finite difference of samples on eval points