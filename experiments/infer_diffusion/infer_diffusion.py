import torch
import joblib
from diffusion.model import TemporalUnet
from diffusion.diffusion import GaussianDiffusion
from diffusion.dataset import TrajectoryDataset


def main():
    domain_size = (10.0, 10.0)
    horizon = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    backbone = TemporalUnet(
        horizon=horizon,
        transition_dim=2,
        belief_dim=1600,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        condition_dropout=0.1,
        kernel_size=5,
    )
    model = GaussianDiffusion(
        backbone,
        horizon=horizon,
        observation_dim=2,
        n_timesteps=100,
        loss_type="l2",
        condition_guidance_w=1.2,
    ).to(device)


    checkpoint_path = "results/train_diffusion/run/2026-03-20_10-55-29/best_model.pt"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()


    dataset_path = "data/trajectories/processed/training_data.pkl"
    dataset = TrajectoryDataset(dataset_path, domain_size)
    sample = dataset[0]

    start_position = sample["tau"][0, :].unsqueeze(0).to(device)  # (1, 2)
    cond = {0: start_position}
    b_mean = sample["b_mean"].unsqueeze(0).to(device)  # (1, 1600)
    b_var = sample["b_var"].unsqueeze(0).to(device)  # (1, 1600)
    r = torch.tensor([[1.0]], device=device)  # (1, 1)

    with torch.no_grad():
        tau_norm = model.conditional_sample(cond, b_mean, b_var, r)
    tau =dataset.denormalise_tau(tau_norm.cpu())
    print("Sampled trajectory (denormalised):")
    print(tau.squeeze(0))
    

if __name__ == "__main__":
    main()
