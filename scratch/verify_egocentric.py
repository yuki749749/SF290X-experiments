import math
import numpy as np
import torch
import torch.nn.functional as F

def test_coordinate_transform():
    print("=== Testing Coordinate Transforms ===")
    
    # 1. Setup positions and heading
    p_curr = np.array([100.0, 100.0], dtype=np.float32)
    heading = math.pi / 4 # 45 degrees
    max_step = 10.0
    domain_size = np.array([300.0, 300.0], dtype=np.float32)
    domain_pad = 50.0
    
    # previous position is max_step behind current position along heading
    p_prev = np.array([
        p_curr[0] - max_step * math.cos(heading),
        p_curr[1] - max_step * math.sin(heading),
    ], dtype=np.float32)
    
    # future position is max_step ahead
    p_next = np.array([
        p_curr[0] + max_step * math.cos(heading),
        p_curr[1] + max_step * math.sin(heading),
    ], dtype=np.float32)
    
    # 2. Local frame transforms (Global -> Body)
    tau_raw = torch.tensor(np.array([p_prev, p_curr, p_next]), dtype=torch.float32)
    
    p_curr_tensor = torch.tensor(p_curr, dtype=torch.float32)
    tau_trans = tau_raw - p_curr_tensor
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    rot_mat_T = torch.tensor([
        [cos_h, sin_h],
        [-sin_h, cos_h]
    ], dtype=torch.float32)
    tau_rot = tau_trans @ rot_mat_T.t()
    
    scale_factor = torch.tensor(2.0 / (domain_size + 2 * domain_pad), dtype=torch.float32)
    tau_body = tau_rot * scale_factor
    
    v_norm_expected = max_step * scale_factor[0].item()
    print(f"p_prev in body frame: {tau_body[0].numpy()} (Expected: [{-v_norm_expected:.4f}, 0.0])")
    print(f"p_curr in body frame: {tau_body[1].numpy()} (Expected: [0.0, 0.0])")
    print(f"p_next in body frame: {tau_body[2].numpy()} (Expected: [{v_norm_expected:.4f}, 0.0])")
    
    assert np.allclose(tau_body[0].numpy(), [-v_norm_expected, 0.0], atol=1e-5)
    assert np.allclose(tau_body[1].numpy(), [0.0, 0.0], atol=1e-5)
    assert np.allclose(tau_body[2].numpy(), [v_norm_expected, 0.0], atol=1e-5)
    
    # 3. Inverse transform (Body -> Global)
    rot_mat = torch.tensor([
        [cos_h, -sin_h],
        [sin_h, cos_h]
    ], dtype=torch.float32)
    tau_scaled = tau_body / scale_factor
    tau_global_reconstructed = tau_scaled @ rot_mat.t() + p_curr_tensor
    
    print(f"Original global positions: {tau_raw.numpy().tolist()}")
    print(f"Reconstructed global positions: {tau_global_reconstructed.numpy().tolist()}")
    assert np.allclose(tau_raw.numpy(), tau_global_reconstructed.numpy(), atol=1e-5)
    print("=> Coordinate transforms passed successfully!")

def test_crop_rotation():
    print("\n=== Testing Crop Rotation ===")
    crop_size = 5
    # Create a 2D map with a single hot spot at the "top" cell (index 0, 2)
    # in PyTorch format: (C, H, W)
    grid_img = torch.zeros(2, crop_size, crop_size, dtype=torch.float32)
    grid_img[0, 0, 2] = 10.0 # Hot spot at top-middle of mean map
    grid_img[1, 0, 2] = 5.0  # Hot spot at top-middle of var map
    
    # Rotate by pi/2 (90 degrees). The hot spot at West (index 0, 2) should rotate to egocentric left (index 2, 4)
    angle = math.pi / 2
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    rot_mat = torch.tensor([[
        [cos_a,  sin_a, 0.0],
        [-sin_a, cos_a, 0.0]
    ]], dtype=torch.float32)
    
    x_batch = grid_img.unsqueeze(0)
    grid = F.affine_grid(rot_mat, x_batch.size(), align_corners=True)
    rotated = F.grid_sample(x_batch, grid, align_corners=True, mode="bilinear", padding_mode="zeros").squeeze(0)
    
    print("Original mean map:")
    print(grid_img[0].numpy())
    print("Rotated (90 deg) mean map:")
    print(rotated[0].numpy())
    
    # The rotated hot spot should be at egocentric left (2, 4)
    assert rotated[0, 2, 4].item() > 5.0
    print("=> Crop rotation passed successfully!")

if __name__ == "__main__":
    test_coordinate_transform()
    test_crop_rotation()
