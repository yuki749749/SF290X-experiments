import random
import torch
import numpy as np


class BasePlanner:
    def __init__(
        self,
        domain_size,
        domain_pad=5.0,
        max_step=1.0,
        min_step=0.5,
        max_turn=np.pi / 8,
        boundary_behavior="reflect",
    ):
        assert 0.0 <= min_step <= max_step, "min_step must be in [0, max_step]"
        self.domain_size = domain_size
        self.domain_pad = domain_pad
        self.max_step = max_step
        self.min_step = min_step
        self.max_turn = max_turn
        if boundary_behavior == "reflect":
            self.handle_boundary = self._reflect
        elif boundary_behavior == "clamp":
            self.handle_boundary = self._clamp
        else:
            raise ValueError(f"Invalid boundary behavior: {boundary_behavior}")

    def compute_next_pose(self, current_position, current_heading, current_belief):
        raise NotImplementedError("Must be implemented by subclass")

    def _reflect(self, position, heading):
        x, y = position
        dx, dy = np.cos(heading), np.sin(heading)
        if not (0 <= x <= self.domain_size[0]):
            dx = -dx
        if not (0 <= y <= self.domain_size[1]):
            dy = -dy
        x = np.clip(x, 0, self.domain_size[0])
        y = np.clip(y, 0, self.domain_size[1])
        return (x, y), np.arctan2(dy, dx)

    def _clamp(self, position, heading):
        x, y = position
        dx, dy = np.cos(heading), np.sin(heading)

        if not (0 <= x <= self.domain_size[0]):
            dx = 0.0  # kill normal component, preserve tangential
        if not (0 <= y <= self.domain_size[1]):
            dy = 0.0

        x = np.clip(x, 0, self.domain_size[0])
        y = np.clip(y, 0, self.domain_size[1])

        # If both components zeroed (hit a corner head-on), reverse heading
        if dx == 0.0 and dy == 0.0:
            dx, dy = -np.cos(heading), -np.sin(heading)

        return (x, y), np.arctan2(dy, dx)
    
    def steer_back_heading(self, position, heading):
        W, H = self.domain_size
        cx, cy = W / 2.0, H / 2.0

        dx = cx - position[0]
        dy = cy - position[1]
        target_angle = np.arctan2(dy, dx)
        # Signed shortest angular distance from current heading to target
        heading_error = np.arctan2(
            np.sin(target_angle - heading),
            np.cos(target_angle - heading),
        )
        # Turn at exactly max_turn in the correct direction
        turn = np.sign(heading_error) * self.max_turn
        return heading + turn

    def _clamp_outer(self, position, heading):
        x, y = position
        dx, dy = np.cos(heading), np.sin(heading)

        if not (-self.domain_pad <= x <= self.domain_size[0] + self.domain_pad):
            dx = 0.0
        if not (-self.domain_pad <= y <= self.domain_size[1] + self.domain_pad):
            dy = 0.0
        x = np.clip(x, -self.domain_pad, self.domain_size[0] + self.domain_pad)
        y = np.clip(y, -self.domain_pad, self.domain_size[1] + self.domain_pad)
        
        if dx == 0.0 and dy == 0.0:
            dx, dy = -np.cos(heading), -np.sin(heading)
        return (x, y), np.arctan2(dy, dx)

    def reset(self, **kwargs):
        pass  # stateless planners need no reset


class DummyPlanner(BasePlanner):
    def compute_next_pose(self, current_position, current_heading, current_belief):
        # Move forward in the current heading direction
        new_heading = current_heading
        return (
            (
                current_position[0] + self.max_step * np.cos(new_heading),
                current_position[1] + self.max_step * np.sin(new_heading),
            ),
            new_heading,
        )


# class LawnmowerPlanner(BasePlanner):

#     def __init__(self, domain_size, max_step=1.0, max_turn=np.pi/8, lane_spacing=None):
#         super().__init__(domain_size, max_step, max_turn)
#         self.lane_spacing = lane_spacing if lane_spacing is not None else max_step
#         self._waypoints = self._build_path()
#         self._waypointIndex = 0

#     def _build_path(self):
#         W, H = self.domain_size
#         waypoints = []
#         ys = np.arange(0, H + self.lane_spacing, self.lane_spacing)
#         for i, y in enumerate(ys):
#             xs = np.arange(0, W + self.max_step, self.max_step)
#             xs = np.clip(xs, 0, W)
#             if i % 2 != 0:
#                 xs = xs[::-1]
#             for x in xs:
#                 waypoints.append((float(x), float(y)))
#         return waypoints

#     def compute_next_pose(self, current_position, current_heading, current_belief):
#         if self._waypointIndex < len(self._waypoints) - 1:
#             self._waypointIndex += 1
#         target = self._waypoints[self._waypointIndex]
#         dx = target[0] - current_position[0]
#         dy = target[1] - current_position[1]
#         new_heading = np.arctan2(dy, dx)
#         return (target, new_heading)


class LawnmowerPlanner(BasePlanner):
    """
    Boustrophedon (back-and-forth) coverage planner.
    Pre-computes a full waypoint path at first call and follows it.
    Lane spacing controls coverage density; when spacing equals max_step,
    the path is Nyquist-sampled for a sensor footprint of that width.
    """

    def __init__(
        self,
        domain_size,
        domain_pad=5.0,
        max_step=1.0,
        min_step=0.5,
        max_turn=np.pi,
        boundary_behavior="clamp",
        orientation="horizontal",
        edge_turn_steps=2,
    ):
        super().__init__(domain_size, domain_pad, max_step, min_step, max_turn, boundary_behavior)
        self.orientation = orientation  # 'horizontal' or 'vertical'
        self.edge_turn_steps = edge_turn_steps
        self.lane_spacing = edge_turn_steps * max_step
        self._waypoints = self._build_path()
        self._waypoint_index = 0

    def _build_path(self):
        """Pre-compute the full boustrophedon waypoint list."""
        W, H = self.domain_size
        waypoints = []

        if self.orientation == "horizontal":
            # Sweep along x, step in y
            ys = np.arange(0, H + self.lane_spacing, self.lane_spacing)
            ys = np.clip(ys, 0, H)
            for i, y in enumerate(ys):
                if i % 2 == 0:
                    xs = np.arange(0, W + self.max_step, self.max_step)
                else:
                    xs = np.arange(W, -self.max_step, -self.max_step)
                xs = np.clip(xs, 0, W)
                for x in xs:
                    waypoints.append((float(x), float(y)))
                # At the x-edge, move 2 steps upward before the next lane
                if i < len(ys) - 1:
                    edge_x = float(W) if i % 2 == 0 else 0.0
                    for step in range(1, self.edge_turn_steps + 1):
                        wp_y = min(y + step * self.max_step, H)
                        waypoints.append((edge_x, wp_y))
        else:
            # Sweep along y, step in x
            xs = np.arange(0, W + self.lane_spacing, self.lane_spacing)
            xs = np.clip(xs, 0, W)
            for i, x in enumerate(xs):
                if i % 2 == 0:
                    ys = np.arange(0, H + self.max_step, self.max_step)
                else:
                    ys = np.arange(H, -self.max_step, -self.max_step)
                ys = np.clip(ys, 0, H)
                for y in ys:
                    waypoints.append((float(x), float(y)))
                # At the y-edge, move 2 steps upward (in x) before the next lane
                if i < len(xs) - 1:
                    edge_y = float(H) if i % 2 == 0 else 0.0
                    for step in range(1, self.edge_turn_steps + 1):
                        wp_x = min(x + step * self.max_step, W)
                        waypoints.append((wp_x, edge_y))

        return waypoints

    def compute_next_pose(self, current_position, current_heading, current_belief):
        # Build path lazily on first call
        if self._waypoints is None:
            self._waypoints = self._build_path()
            self._waypoint_index = 0

        # Advance waypoint index if we're close enough to the current target
        while self._waypoint_index < len(self._waypoints) - 1:
            target = self._waypoints[self._waypoint_index]
            dist = np.hypot(
                target[0] - current_position[0], target[1] - current_position[1]
            )
            if dist < self.max_step * 0.5:  # within half a step = "arrived"
                self._waypoint_index += 1
            else:
                break

        target = self._waypoints[self._waypoint_index]
        dx = target[0] - current_position[0]
        dy = target[1] - current_position[1]
        dist = np.hypot(dx, dy)

        # Heading toward waypoint
        desired_heading = np.arctan2(dy, dx)

        # Clamp turn to max_turn (graceful degradation near kinematic limits)
        heading_error = np.arctan2(
            np.sin(desired_heading - current_heading),
            np.cos(desired_heading - current_heading),
        )
        turn = np.clip(heading_error, -self.max_turn, self.max_turn)
        new_heading = current_heading + turn

        # Move up to max_step toward waypoint
        step = min(self.max_step, dist)
        new_position = (
            current_position[0] + step * np.cos(new_heading),
            current_position[1] + step * np.sin(new_heading),
        )

        return (new_position, new_heading)

    def reset(self, **kwargs):
        self._waypoint_index = 0


class RandomPlanner(BasePlanner):
    def __init__(
        self,
        domain_size,
        domain_pad=5.0,
        max_step=1,
        min_step=0.5,
        max_turn=np.pi / 8,
        boundary_behavior="clamp",
    ):
        super().__init__(domain_size, domain_pad, max_step, min_step, max_turn, boundary_behavior)

    def compute_next_pose(self, current_position, current_heading, current_belief):
        x, y = current_position
        W, H = self.domain_size

        if 0 <= x <= W and 0 <= y <= H:
            # In-domain: random step and turn
            turn = random.uniform(-self.max_turn, self.max_turn)
            new_heading = current_heading + turn
            lo = (self.min_step / self.max_step) ** 2
            step = np.sqrt(random.uniform(lo, 1.0)) * self.max_step
            new_position = (
                current_position[0] + step * np.cos(new_heading),
                current_position[1] + step * np.sin(new_heading),
            )
        else:
            # Out-of-domain: steer back toward center
            new_heading = self.steer_back_heading(current_position, current_heading)
            new_position = (
                current_position[0] + self.max_step * np.cos(new_heading),
                current_position[1] + self.max_step * np.sin(new_heading),
            )

        # lo = (self.min_step / self.max_step) ** 2
        # step = np.sqrt(random.uniform(lo, 1.0)) * self.max_step
        # turn = random.uniform(-self.max_turn, self.max_turn)
        # new_heading = current_heading + turn
        # new_position = (
        #     current_position[0] + step * np.cos(new_heading),
        #     current_position[1] + step * np.sin(new_heading),
        # )
        new_position, new_heading = self._clamp_outer(new_position, new_heading)
        return (new_position, new_heading)


class BayesianOptimizationPlanner(BasePlanner):
    def __init__(
        self,
        domain_size,
        domain_pad=5.0,
        max_step=1.0,
        min_step=0.5,
        max_turn=np.pi / 4,
        boundary_behavior="clamp",
        acquisition_function="ucb",
        beta=2.0,
        num_steps=10,
        num_turns=5,
        acq_mask_sharpness=3.0,
        randomize_first_step=False,
    ):
        super().__init__(domain_size, domain_pad, max_step, min_step, max_turn, boundary_behavior)
        self.acquisition_function = acquisition_function
        self.beta = beta
        self.num_steps = num_steps
        self.num_turns = num_turns
        self.acq_mask_sharpness = acq_mask_sharpness
        self.randomize_first_step = randomize_first_step
        self._step = 0

    def compute_next_pose(self, current_position, current_heading, current_belief):
        x, y = current_position
        W, H = self.domain_size
        if 0 <= x <= W and 0 <= y <= H:
            if self.randomize_first_step and self._step == 0:
                turn = random.uniform(-self.max_turn, self.max_turn)
                new_heading = current_heading + turn
                lo = (self.min_step / self.max_step) ** 2
                step = np.sqrt(random.uniform(lo, 1.0)) * self.max_step
                new_position = (
                    current_position[0] + step * np.cos(new_heading),
                    current_position[1] + step * np.sin(new_heading),
                )
            else:
                # In-domain: compute acquisition and pick best candidate
                candidate_position_list, candidate_heading_list = self.generate_candidates(
                    current_position, current_heading
                )
                acquisition_values = self.compute_acquisition(
                    current_belief, candidate_position_list
                )
                best_candidate_position = candidate_position_list[acquisition_values.argmax()]
                best_candidate_heading = candidate_heading_list[acquisition_values.argmax()]
                new_position = tuple(best_candidate_position.tolist())
                new_heading = best_candidate_heading
        else:
            new_heading = self.steer_back_heading(current_position, current_heading)
            step = self.max_step
            new_position = (
                current_position[0] + step * np.cos(new_heading),
                current_position[1] + step * np.sin(new_heading),
            )
        new_position, new_heading = self._clamp_outer(new_position, new_heading)
        self._step += 1
        return (new_position, new_heading)
    
    def _acquisition_mask(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Smooth sigmoid mask that suppresses acquisition outside [0, D]^2.
 
        For each axis i:  phi_i(x) = sigmoid(s * x_i) * sigmoid(s * (D_i - x_i))
        where s = acq_mask_sharpness / max_step controls the falloff width.
 
        Inside the domain phi ~ 1; outside it decays toward 0.
        The two-axis product gives a corner-aware mask.
        """
        s = self.acq_mask_sharpness / self.max_step
        W, H = self.domain_size
        x = candidates[:, 0]
        y = candidates[:, 1]
        phi_x = torch.sigmoid(s * x) * torch.sigmoid(s * (W - x))
        phi_y = torch.sigmoid(s * y) * torch.sigmoid(s * (H - y))
        # Rescale so mask == 1 at domain centre (optional, keeps UCB scale intact)
        centre_val = torch.sigmoid(torch.tensor(s * W / 2)) ** 2
        return (phi_x * phi_y) / (centre_val ** 2)

    def compute_acquisition(self, current_belief, candidates):
        means, variance = current_belief.predict(candidates)
        stds = variance.sqrt()
        if self.acquisition_function == "ucb":
            return means + self.beta * stds
        elif self.acquisition_function == "variance":
            return stds
        else:
            raise ValueError(
                f"Unknown acquisition function: {self.acquisition_function}"
            )

    def generate_candidates(
        self, current_position, current_heading
    ):  # Add AUV dynamics constraints here later
        # Generate random candidates around current position
        candidate_position_list = []
        candidate_heading_list = []

        step_list = np.sqrt(
            np.linspace(self.min_step**2, self.max_step**2, self.num_steps)
        )
        turn_list = np.linspace(-self.max_turn, self.max_turn, self.num_turns)
        for step in step_list:
            for turn in turn_list:
                new_heading = current_heading + turn
                new_position = (
                    current_position[0] + step * np.cos(new_heading),
                    current_position[1] + step * np.sin(new_heading),
                )
                candidate_position_list.append(new_position)
                candidate_heading_list.append(new_heading)

        if (
            len(candidate_position_list) == 0
        ):  # If no valid candidates found, stay in place
            candidate_position_list.append(current_position)
            candidate_heading_list.append(current_heading)

        return torch.tensor(
            candidate_position_list, dtype=torch.float32
        ), candidate_heading_list

    def reset(self, **kwargs):
        self._step = 0
