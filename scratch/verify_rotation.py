import torch
import torch.nn.functional as F
import math
import numpy as np

def crop_belief_map(
    b_mean: torch.Tensor,       # (grid_size²,) flattened
    b_var: torch.Tensor,        # (grid_size²,) flattened
    boundary_map: torch.Tensor, # (grid_size²,) flattened
    agent_pos: torch.Tensor,    # (2,) physical coordinates
    domain_min: torch.Tensor,   # (2,)
    domain_max: torch.Tensor,   # (2,)
    crop_size: int,
    grid_size: int = 40,
):
    norm = (agent_pos - domain_min) / (domain_max - domain_min)  # [0, 1]
    ci = int(round(norm[0].item() * (grid_size - 1)))
    cj = int(round(norm[1].item() * (grid_size - 1)))

    half = crop_size // 2
    src_i0, src_i1 = ci - half, ci - half + crop_size
    src_j0, src_j1 = cj - half, cj - half + crop_size

    si0 = max(0, min(grid_size, src_i0))
    si1 = max(0, min(grid_size, src_i1))
    sj0 = max(0, min(grid_size, src_j0))
    sj1 = max(0, min(grid_size, src_j1))

    dst_i0 = max(0, -src_i0)
    dst_i1 = dst_i0 + (si1 - si0)
    dst_j0 = max(0, -src_j0)
    dst_j1 = dst_j0 + (sj1 - sj0)

    mean_map = b_mean.view(grid_size, grid_size)
    var_map  = b_var.view(grid_size, grid_size)
    bound_map = boundary_map.view(grid_size, grid_size)

    mean_crop = torch.zeros(crop_size, crop_size, dtype=b_mean.dtype, device=b_mean.device)
    var_crop  = torch.ones(crop_size, crop_size, dtype=b_var.dtype,  device=b_var.device)
    bound_crop = torch.ones(crop_size, crop_size, dtype=boundary_map.dtype, device=boundary_map.device)

    if si1 > si0 and sj1 > sj0:
        mean_crop[dst_i0:dst_i1, dst_j0:dst_j1] = mean_map[si0:si1, sj0:sj1]
        var_crop[dst_i0:dst_i1,  dst_j0:dst_j1] = var_map[si0:si1,  sj0:sj1]
        bound_crop[dst_i0:dst_i1, dst_j0:dst_j1] = bound_map[si0:si1, sj0:sj1]

    return mean_crop, var_crop, bound_crop

def test():
    # Setup domains
    domain_size = (300.0, 300.0)
    domain_pad = 50.0
    grid_size = 40
    crop_size = 11
    
    domain_min = torch.tensor([-domain_pad, -domain_pad])
    domain_max = torch.tensor([domain_size[0] + domain_pad, domain_size[1] + domain_pad])
    
    # Let's place the agent at the center: physical (150, 150)
    agent_pos = torch.tensor([150.0, 150.0])
    
    # We create a belief map where there is a high plume source (value 1.0)
    # physically ahead of the agent.
    # Let's say heading = pi / 2 (pointing along +y axis physically).
    # So physically, "ahead" means pointing along +y.
    # Let's place the source at physical x = 150.0, physical y = 200.0 (+50m ahead).
    
    b_mean = torch.zeros(grid_size, grid_size)
    boundary_map = torch.zeros(grid_size, grid_size)
    
    # Convert physical coordinates to grid indices
    # (x - domain_min) / (domain_max - domain_min) * (grid_size - 1)
    def to_grid_idx(x, y):
        gi = int(round(((x - domain_min[0].item()) / (domain_max[0].item() - domain_min[0].item())) * (grid_size - 1)))
        gj = int(round(((y - domain_min[1].item()) / (domain_max[1].item() - domain_min[1].item())) * (grid_size - 1)))
        return gi, gj
    
    si, sj = to_grid_idx(150.0, 200.0)
    b_mean[si, sj] = 9.9 # mark the source
    
    # Let's verify agent pos grid coordinates
    ai, aj = to_grid_idx(150.0, 150.0)
    print(f"Agent Grid Index: ({ai}, {aj})")
    print(f"Source Grid Index: ({si}, {sj})")
    print(f"Source is at index delta: ({si - ai}, {sj - aj})")
    
    # Crop
    mean_crop, var_crop, bound_crop = crop_belief_map(
        b_mean.flatten(), torch.zeros_like(b_mean).flatten(), boundary_map.flatten(),
        agent_pos, domain_min, domain_max, crop_size, grid_size
    )
    
    # Let's check crop before rotation
    # Crop center is at index half = 5.
    half = crop_size // 2
    # The source should be at crop index (half + (si - ai), half + (sj - aj)) = (5, 5 + 5) = (5, 10).
    print("\n--- Crop Before Rotation ---")
    for r in range(crop_size):
        row_str = " ".join(f"{mean_crop[r, c].item():.1f}" for c in range(crop_size))
        print(row_str)
        
    # Now let's apply egocentric rotation
    # Heading = pi / 2
    heading = math.pi / 2
    cos_a = math.cos(heading)
    sin_a = math.sin(heading)
    
    # Rotation matrix in PyTorch F.affine_grid
    rot_mat = torch.tensor([[
        [cos_a,  sin_a, 0.0],
        [-sin_a, cos_a, 0.0]
    ]], dtype=torch.float32)
    
    grid_img = torch.stack([mean_crop, var_crop, bound_crop], dim=0) # (3, H, W)
    x_batch = grid_img.unsqueeze(0)
    grid = F.affine_grid(rot_mat, x_batch.size(), align_corners=True)
    rotated = F.grid_sample(x_batch, grid, align_corners=True, mode="bilinear", padding_mode="zeros")
    rotated = rotated.squeeze(0)
    
    rotated_mean = rotated[0]
    print("\n--- Crop After Rotation (heading = pi/2) ---")
    for r in range(crop_size):
        row_str = " ".join(f"{rotated_mean[r, c].item():.1f}" for c in range(crop_size))
        print(row_str)
        
    # In egocentric coordinates:
    # A point ahead of the AUV (along its heading) should be mapped to the egocentric forward direction.
    # What is the forward direction of the trajectory in egocentric frame?
    # Let's rotate the physical delta (dx = 0, dy = 50.0):
    tau_trans = torch.tensor([[0.0, 50.0]], dtype=torch.float32)
    rot_mat_tau = torch.tensor([
        [cos_a, sin_a],
        [-sin_a, cos_a]
    ], dtype=torch.float32)
    tau_rot = tau_trans @ rot_mat_tau.t()
    print(f"\nRotated Trajectory Point: {tau_rot.numpy()}")
    # Rotated trajectory point is [50.0, 0.0].
    # That means:
    # Index 0 (egocentric x) is forward = +50.0
    # Index 1 (egocentric y) is lateral = 0.0
    
    # Now let's look at the rotated crop.
    # Where does the source (value 9.9) appear in the rotated crop?
    # Does it appear along the first dimension (row index = egocentric x) or second (col index = egocentric y)?
    # Let's find its index in rotated_mean:
    max_idx = torch.argmax(rotated_mean)
    ri, rj = max_idx // crop_size, max_idx % crop_size
    print(f"Source in Rotated Crop at index: ({ri.item()}, {rj.item()})")
    print(f"Relative to center ({half}, {half}): ({ri.item() - half}, {rj.item() - half})")
    
if __name__ == "__main__":
    test()
