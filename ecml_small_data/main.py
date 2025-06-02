import copy
import dill
import gpytorch
import itertools
from lodegp import LODEGP
from helpers.training_functions import granso_optimization
from helpers.plotting_functions import plot_single_input_gp_posterior
from helpers.util_functions import central_difference, calculate_differential_equation_error_numeric, calculate_differential_equation_error_symbolic
from sage.all import *
import sage
from sklearn.metrics import mean_squared_error
import torch

torch.set_default_dtype(torch.float64)

torch.manual_seed(42)

R = QQ['x']; (x,) = R._first_ngens(1)
true_system_description = matrix(R, Integer(2), Integer(3), [x**2 + 9.81/1.0, 0, -1/1.0, 0, x**2+9.81/2.0, -1/2.0])

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
        
        # Create the experiment directory
        experiment_dir_addendum = f"{START}-{END}-{COUNT}_{noise_level}"
        base_experiment_dir = f"./results"
        base_figures_dir = os.path.join(base_experiment_dir, "figures")
        base_results_dir = os.path.join(base_experiment_dir, "results")

        experiment_figures_dir = os.path.join(base_figures_dir, experiment_dir_addendum)
        experiment_results_dir = os.path.join(base_results_dir, experiment_dir_addendum)

        if not os.path.exists(experiment_figures_dir):
            os.makedirs(experiment_figures_dir)
        if not os.path.exists(experiment_results_dir):
            os.makedirs(experiment_results_dir)
        
        filename_addendum = f"{system_name}_l1-{l1_param_val}_l2-{l2_param_val}"

        # What results are stored?
        # - MLL and MAP loss
        # - Training logs
        # - ODE satisfaction (with varying h-step)
        # - MWE to the train data
        # - MWE to the test data
        # - GP samples ODE satisfaction
        # - Plot GP samples
        # - Plot GP posterior


        # 1. Model definition
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
        model = LODEGP.LODEGP(train_x, train_y, likelihood, 3, ODE_name=system_name, verbose=False, system_parameters={"l1": l1_param_val, "l2": l2_param_val})

        likelihood_MAP = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
        model_MAP = LODEGP.LODEGP(train_x, train_y, likelihood_MAP, 3, ODE_name=system_name, verbose=False, system_parameters={"l1": l1_param_val, "l2": l2_param_val})


        # 2. Model training (10 random restarts with PyGRANSO) (try MLL and MAP with uninformed prior)
        model.train()
        likelihood.train()
        unscaled_neg_MLL, model, likelihood, training_logs = granso_optimization(model, likelihood, train_x, train_y, random_restarts=10, uninformed=True, logarithmic_reinit=True, verbose=False, MAP=False, maxit=300)
        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MLL.pkl"), "wb") as f:
            dill.dump(unscaled_neg_MLL, f)
        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MLL_logs.pkl"), "wb") as f:
            dill.dump(training_logs, f)
        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MLL_state_dict.pkl"), "wb") as f:
            dill.dump(copy.deepcopy(model.state_dict()), f)
        

        # print model parameters
        print("Model parameters after training:")
        print(list(model.named_parameters()))

        model_MAP.train()
        likelihood_MAP.train()
        unscaled_neg_MAP, model_MAP, likelihood_MAP, training_logs_MAP = granso_optimization(model_MAP, likelihood_MAP, train_x, train_y, random_restarts=10, uninformed=True, logarithmic_reinit=True, verbose=False, MAP=True, maxit=300)

        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MAP.pkl"), "wb") as f:
            dill.dump(unscaled_neg_MAP, f)
        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MAP_logs.pkl"), "wb") as f:
            dill.dump(training_logs_MAP, f)
        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MAP_state_dict.pkl"), "wb") as f:
            dill.dump(copy.deepcopy(model_MAP.state_dict()), f)

        print("Model parameters after training:")
        print(list(model_MAP.named_parameters()))

        # 3. Draw model posterior
        test_x = torch.linspace(1e-3, 15, 100)
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

        y0_test = y0_func(test_x)
        y1_test = y1_func(test_x)
        y2_test = y2_func(test_x)

        test_y = torch.stack([y0_test, y1_test, y2_test], dim=-1)

        MLL_model_posterior_ax[0].plot(test_x, y0_test, color="red", linestyle="--") 
        MLL_model_posterior_ax[1].plot(test_x, y1_test, color="red", linestyle="--")
        MLL_model_posterior_ax[2].plot(test_x, y2_test, color="red", linestyle="--") 

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
        MAP_model_posterior_ax[0].plot(test_x, y0_test, color="red", linestyle="--") 
        MAP_model_posterior_ax[1].plot(test_x, y1_test, color="red", linestyle="--")
        MAP_model_posterior_ax[2].plot(test_x, y2_test, color="red", linestyle="--") 

        # Store the figures
        MLL_model_posterior_fig.savefig(f"{experiment_figures_dir}/MLL_model_posterior_{system_name}_l1-{l1_param_val}_l2-{l2_param_val}_{START}-{END}-{COUNT}_{noise_level}.png")
        MAP_model_posterior_fig.savefig(f"{experiment_figures_dir}/MAP_model_posterior_{system_name}_l1-{l1_param_val}_l2-{l2_param_val}_{START}-{END}-{COUNT}_{noise_level}.png")

        # 4. Calculate various metrics
        # 4.1 ODE satisfaction
        # Verify that the models output satisfies the given differential equation
        for target_row in range(len(true_system_description.rows())):
            # Eval mode because central_difference is used
            model.eval()
            likelihood.eval()
            model_mean_generator = lambda x: model(x).mean
            locals_values = model.prepare_numeric_ode_satisfaction_check()
            mean_ode_satisfaction_error_MLL = calculate_differential_equation_error_numeric(true_system_description[target_row], model.sage_locals, model_mean_generator, test_x, locals_values=locals_values)

            model_MAP.eval()
            likelihood_MAP.eval()
            MAP_locals_values = model_MAP.prepare_numeric_ode_satisfaction_check()
            model_MAP_mean_generator = lambda x: model_MAP(x).mean
            MAP_locals_values = model.prepare_numeric_ode_satisfaction_check()
            mean_ode_satisfaction_error_MAP = calculate_differential_equation_error_numeric(true_system_description[target_row], model_MAP.sage_locals, model_MAP_mean_generator, test_x, locals_values=MAP_locals_values)

            with open(os.path.join(experiment_results_dir, f"{filename_addendum}_mean_ode_satisfaction_error_MLL.pkl"), "wb") as f:
                dill.dump(mean_ode_satisfaction_error_MLL, f)
            with open(os.path.join(experiment_results_dir, f"{filename_addendum}_mean_ode_satisfaction_error_MAP.pkl"), "wb") as f:
                dill.dump(mean_ode_satisfaction_error_MAP, f)
        # 4.2 GP samples ODE satisfaction
            model_sample_generator = lambda x: likelihood(model(x)).sample()
            sample_ode_satisfaction_error_MLL = calculate_differential_equation_error_numeric(true_system_description[target_row], model.sage_locals, model_sample_generator, test_x, locals_values=locals_values)

            model_MAP_sample_generator = lambda x: likelihood_MAP(model_MAP(x)).sample()
            sample_ode_satisfaction_error_MAP = calculate_differential_equation_error_numeric(true_system_description[target_row], model_MAP.sage_locals, model_MAP_sample_generator, test_x, locals_values=MAP_locals_values)

            with open(os.path.join(experiment_results_dir, f"{filename_addendum}_sample_ode_satisfaction_error_MLL.pkl"), "wb") as f:
                dill.dump(sample_ode_satisfaction_error_MLL, f)
            with open(os.path.join(experiment_results_dir, f"{filename_addendum}_sample_ode_satisfaction_error_MAP.pkl"), "wb") as f:
                dill.dump(sample_ode_satisfaction_error_MAP, f)



        # 4.3 MSE to train data

        MLL_model_train_MSEs = [mean_squared_error(train_y[:, 0].detach(), model(train_x).mean[:, 0].detach()), mean_squared_error(train_y[:, 1].detach(), model(train_x).mean[:, 1].detach()), mean_squared_error(train_y[:, 2].detach(), model(train_x).mean[:, 2].detach())]
        MAP_model_train_MSEs = [mean_squared_error(train_y[:, 0].detach(), model_MAP(train_x).mean[:, 0].detach()), mean_squared_error(train_y[:, 1].detach(), model_MAP(train_x).mean[:, 1].detach()), mean_squared_error(train_y[:, 2].detach(), model_MAP(train_x).mean[:, 2].detach())]
        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MLL_model_train_MSEs.pkl"), "wb") as f:
            dill.dump(MLL_model_train_MSEs, f)
        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MAP_model_train_MSEs.pkl"), "wb") as f:
            dill.dump(MAP_model_train_MSEs, f)

        # 4.4 MSE to test data
        MLL_model_test_MSEs = [mean_squared_error(test_y[:, 0].detach(), model(test_x).mean[:, 0].detach()), mean_squared_error(test_y[:, 1].detach(), model(test_x).mean[:, 1].detach()), mean_squared_error(test_y[:, 2].detach(), model(test_x).mean[:, 2].detach())]
        MAP_model_test_MSEs = [mean_squared_error(test_y[:, 0].detach(), model_MAP(test_x).mean[:, 0].detach()), mean_squared_error(test_y[:, 1].detach(), model_MAP(test_x).mean[:, 1].detach()), mean_squared_error(test_y[:, 2].detach(), model_MAP(test_x).mean[:, 2].detach())]
        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MLL_model_test_MSEs.pkl"), "wb") as f:
            dill.dump(MLL_model_test_MSEs, f)
        with open(os.path.join(experiment_results_dir, f"{filename_addendum}_MAP_model_test_MSEs.pkl"), "wb") as f:
            dill.dump(MAP_model_test_MSEs, f)
