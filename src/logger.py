import numpy as np
import torch
import gpytorch
import os
import joblib

class Logger:
    def __init__(self, output_directory, eval_x, vis_x, ground_truth_eval, ground_truth_vis=None):
        self.output_directory = output_directory
        self.eval_x = eval_x
        self.vis_x = vis_x
        self.ground_truth_eval = ground_truth_eval
        self.ground_truth_vis = ground_truth_vis
        self.position_history = []
        self.mean_history = []
        self.variance_history = []
        self.rmse_history = []
        # self.nlpd_history = []
        self.normalized_trace_reduction_history = []
        self.initial_trace = None

    def log_step(self, position, belief):
        mean_vis, var_vis = belief.predict(self.vis_x)
        mean_eval, var_eval = belief.predict(self.eval_x)
        
        if self.initial_trace is None:
            self.initial_trace = var_eval.sum().item()
        self.position_history.append(position)
        self.mean_history.append(mean_vis)
        self.variance_history.append(var_vis)

        rmse = self.compute_rmse(mean_eval)
        self.rmse_history.append(rmse)
        # nlpd = self.computeNLPD(belief, self.ground_truth)
        # self.nlpd_history.append(nlpd)
        self.normalized_trace_reduction = self.compute_normalized_trace_reduction(var_eval)
        self.normalized_trace_reduction_history.append(self.normalized_trace_reduction)

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
            'normalized_trace_reduction_history': self.normalized_trace_reduction_history
        }, output_path)

    def compute_rmse(self, mean):
        f_max = self.ground_truth_eval.max().item()
        f_min = self.ground_truth_eval.min().item()

        # return gpytorch.metrics.mean_squared_error(mean, self.ground_truth_eval, squared=False).item() / (fMax - fMin)
        return torch.square(mean - self.ground_truth_eval).mean().item() ** 0.5 / (f_max - f_min)

    # def computeNLPD(self, belief, ground_truth):
    #     return gpytorch.metrics.negative_log_predictive_density(belief.likelihood(), ground_truth).item()
    
    
    def compute_normalized_trace_reduction(self, variance):
        current_trace = variance.sum().item()
        return (self.initial_trace - current_trace) / self.initial_trace
