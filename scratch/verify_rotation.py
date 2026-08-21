import torch
import torch.nn.functional as F
import math

def test_rotation():
    # Create a 5x5 grid with a single distinct marker to track position
    # Let row be x (0 to 4), col be y (0 to 4).
    # Place a marker at row=1, col=2 (x=1, y=2)
    grid_img = torch.zeros((1, 2, 5, 5))
    grid_img[0, 0, 1, 2] = 1.0  # Channel 0 has marker at (1, 2)
    
    # We want to rotate the grid by angle a = pi/2 (90 degrees counter-clockwise)
    # Under a 90 degree CCW rotation, the point (x, y) should map:
    # If we rotate the coordinate system by +pi/2, the new axes are:
    # x_new = y, y_new = -x
    # So the point (1, 2) should become (2, -1) in the new coordinate system.
    # If we rotate the image by +pi/2 (CCW), the point at (1, 2) moves to (-2, 1).
    
    heading = math.pi / 2
    cos_a = math.cos(heading)
    sin_a = math.sin(heading)
    
    # Let's apply the inverted rot_mat:
    rot_mat = torch.tensor([[
        [cos_a,  sin_a, 0.0],
        [-sin_a, cos_a, 0.0]
    ]], dtype=torch.float32)
    
    grid = F.affine_grid(rot_mat, grid_img.size(), align_corners=True)
    rotated = F.grid_sample(grid_img, grid, align_corners=True, mode="nearest", padding_mode="zeros")
    
    print("Original grid (Channel 0):")
    print(grid_img[0, 0])
    print("Rotated grid (Channel 0):")
    print(rotated[0, 0])

if __name__ == "__main__":
    test_rotation()
