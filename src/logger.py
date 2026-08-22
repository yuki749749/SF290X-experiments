import numpy as np
import torch
import gpytorch
import os
import joblib

class Logger:
    def __init__(
        self,
        output_directory,
        eval_x,
        vis_x,
        ground_truth_eval,
        ground_truth_vis=None,
        stride=None,
        horizon=None,
        n_timesteps=None,
    ):
        self.output_directory = output_directory
        self.eval_x = eval_x
        self.vis_x = vis_x
        self.ground_truth_eval = ground_truth_eval
        self.ground_truth_vis = ground_truth_vis
        self.stride = stride
        self.horizon = horizon
        self.n_timesteps = n_timesteps
        self.position_history = []
        self.mean_history = []
        self.variance_history = []
        self.rmse_history = []
        # self.nlpd_history = []

    def log_step(self, position, belief):
        self.position_history.append(position)
        idx = len(self.position_history) - 1

        if self.stride is None:
            need_pred = True
        else:
            horizon = self.horizon if self.horizon is not None else 16
            n_timesteps = self.n_timesteps if self.n_timesteps is not None else 400
            
            # Compute GP predictions only at indices that:
            # - start a window: idx % stride == 0
            # - end a window (for reward calculation): (idx - (horizon - 3)) % stride == 0
            # - represent the midpoint (for midpoint visualization): idx == n_timesteps // 2
            # - represent the final step (for metrics): idx == n_timesteps
            need_pred = (
                (idx % self.stride == 0) or
                ((idx - (horizon - 3)) % self.stride == 0) or
                (idx == n_timesteps) or
                (idx == n_timesteps // 2)
            )

        if need_pred:
            mean_vis, var_vis = belief.predict(self.vis_x)
            mean_eval, var_eval = belief.predict(self.eval_x)
            rmse = self.compute_rmse(mean_eval)
        else:
            mean_vis = torch.zeros(self.vis_x.shape[0], dtype=torch.float32)
            var_vis = torch.zeros(self.vis_x.shape[0], dtype=torch.float32)
            rmse = 0.0

        self.mean_history.append(mean_vis)
        self.variance_history.append(var_vis)
        self.rmse_history.append(rmse)
        # nlpd = self.computeNLPD(belief, self.ground_truth)
        # self.nlpd_history.append(nlpd)


    def save_history(self, filename):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        output_path = os.path.join(self.output_directory, filename)
        joblib.dump({
            # 'ground_truth_eval': self.ground_truth_eval,
            'ground_truth': self.ground_truth_vis,
            # 'eval_x': self.eval_x,
            # 'vis_x': self.vis_x,
            'position_history': self.position_history,
            'mean_history': self.mean_history,
            'variance_history': self.variance_history,
            'rmse_history': self.rmse_history,
            # 'nlpd_history': self.nlpd_history,
        }, output_path)

    def compute_rmse(self, mean):
        f_max = self.ground_truth_eval.max().item()
        f_min = self.ground_truth_eval.min().item()

        # return gpytorch.metrics.mean_squared_error(mean, self.ground_truth_eval, squared=False).item() / (fMax - fMin)
        return torch.square(mean - self.ground_truth_eval).mean().item() ** 0.5 / (f_max - f_min)

    # def computeNLPD(self, belief, ground_truth):
    #     return gpytorch.metrics.negative_log_predictive_density(belief.likelihood(), ground_truth).item()

