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




class SimpleMaternGP:
    def __init__(self, mean_constant, lengthscale, outputscale, noise, train_x, train_y):
        self.mean_constant = torch.tensor(mean_constant, dtype=torch.float32)
        self.lengthscale = torch.tensor(lengthscale, dtype=torch.float32)
        self.outputscale = torch.tensor(outputscale, dtype=torch.float32)
        self.noise = torch.tensor(noise, dtype=torch.float32)
        self.train_x = train_x.clone().detach().float()
        self.train_y = train_y.clone().detach().float()
        self._L = None
        self._alpha = None

    def update(self, new_x, new_y):
        self.train_x = torch.cat([self.train_x, new_x.to(self.train_x.device).float()], dim=0)
        self.train_y = torch.cat([self.train_y, new_y.to(self.train_y.device).float()], dim=0)
        self._L = None
        self._alpha = None

    def _matern_kernel(self, x1, x2):
        dev = self.train_x.device
        x1 = x1.to(dev)
        x2 = x2.to(dev)
        lscale = self.lengthscale.to(dev)
        opscale = self.outputscale.to(dev)
        x1_scaled = x1 / lscale
        x2_scaled = x2 / lscale
        dist = torch.cdist(x1_scaled, x2_scaled, p=2.0)
        sqrt5 = 2.23606797749979
        val = 1.0 + sqrt5 * dist + (5.0 / 3.0) * (dist ** 2)
        return opscale * val * torch.exp(-sqrt5 * dist)

    def predict(self, test_x):
        orig_device = test_x.device
        dev = self.train_x.device
        test_x = test_x.to(dev)
        N = self.train_x.size(0)
        if self._L is None or self._alpha is None:
            K_XX = self._matern_kernel(self.train_x, self.train_x)
            noise_t = self.noise.to(dev)
            K_XX_noisy = K_XX + noise_t * torch.eye(N, dtype=torch.float32, device=dev)
            m_const = self.mean_constant.to(dev)
            y_centered = self.train_y - m_const

            self._L = torch.linalg.cholesky(K_XX_noisy)
            w = torch.linalg.solve_triangular(self._L, y_centered.unsqueeze(-1), upper=False)
            self._alpha = torch.linalg.solve_triangular(self._L.t(), w, upper=True).squeeze(-1)

        K_star_X = self._matern_kernel(test_x, self.train_x)
        m_const = self.mean_constant.to(dev)
        pred_mean = K_star_X.mv(self._alpha) + m_const

        v = torch.linalg.solve_triangular(self._L, K_star_X.t(), upper=False)
        opscale = self.outputscale.to(dev)
        pred_var = opscale - torch.sum(v ** 2, dim=0)
        pred_var = torch.clamp(pred_var, min=1e-8)

        return pred_mean.to(orig_device), pred_var.to(orig_device)


class Belief:
    def __init__(self, model, eval_x):
        self.model = model
        self.eval_x = eval_x
        if hasattr(self.model, "eval"):
            self.model.eval()

    def update(self, position, measurement):
        if position.dim() == 1:
            position = position.unsqueeze(0)
        if measurement.dim() == 0:
            measurement = measurement.unsqueeze(0)

        if hasattr(self.model, "get_fantasy_model"):
            self.model = self.model.get_fantasy_model(position, measurement)
        else:
            self.model.update(position, measurement)

    def predict(self, eval_x):
        if hasattr(self.model, "predict"):
            return self.model.predict(eval_x)
        else:
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

    
