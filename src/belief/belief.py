import gpytorch
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale_prior=None):
        if train_x.dim() == 1:
            train_x = train_x.unsqueeze(0)
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(ard_num_dims=2)
        #     )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
        )
        if lengthscale_prior is None:
            lengthscale_prior = gpytorch.priors.GammaPrior(6.0, 3.0)
        self.covar_module.register_prior(
            "lengthscale_prior",
            lengthscale_prior,
            lambda module: module.base_kernel.lengthscale
        )
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)




class Belief:
    def __init__(self, model, eval_x):
        self.model = model.eval()
        self.eval_x = eval_x

    def update(self, position, measurement):
        if position.dim() == 1:
            position = position.unsqueeze(0)
        if measurement.dim() == 0:
            measurement = measurement.unsqueeze(0)
        self.model = self.model.get_fantasy_model(position, measurement)

    def predict(self, eval_x):
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.max_root_decomposition_size(50):
            posterior = self.model(eval_x)
            return posterior.mean, posterior.variance
    
    def predict_eval_x(self):
        return self.predict(self.eval_x)
    
    # def posterior(self):
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.max_root_decomposition_size(50):
    #         posterior = self.model(self.testGrid)
    #         return posterior
        
    # def likelihood(self):
    #     with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples(), gpytorch.settings.max_root_decomposition_size(50):
    #         return self.model.likelihood(self.model(self.testGrid))

    
