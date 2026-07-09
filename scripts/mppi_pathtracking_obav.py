import math
import numpy as np
from typing import Tuple
from pathtracking_kbm_obav import Vehicle

class MPPIControllerForPathTracking():
    def __init__(
            self,
            delta_t: float = 0.05,
            wheel_base: float = 2.5, # [m]
            vehicle_width: float = 3.0, # [m]
            vehicle_length: float = 4.0, # [m]
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]),
            stage_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            visualize_optimal_traj = True,  # if True, optimal trajectory is visualized
            visualze_sampled_trajs = False, # if True, sampled trajectories are visualized
            obstacle_circles: np.ndarray = np.array([[-2.0, 1.0, 1.0], [2.0, -1.0, 1.0]]), # [obs_x, obs_y, obs_radius]
            collision_safety_margin_rate: float = 1.2, # safety margin for collision check
    ) -> None:
        """initialize mppi controller for path-tracking"""
        # mppi parameters
        self.dim_x = 4 # dimension of system state vector
        self.dim_u = 2 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))
        self.Sigma = sigma
        self.Sigma_inv = np.linalg.inv(sigma)  # precompute inverse
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

        # vehicle parameters
        self.delta_t = delta_t
        self.wheel_base = wheel_base
        self.vehicle_w = vehicle_width
        self.vehicle_l = vehicle_length
        self.max_steer_abs = max_steer_abs
        self.max_accel_abs = max_accel_abs
        self.ref_path = ref_path

        # obstacle parameters
        self.obstacle_circles = obstacle_circles
        self.collision_safety_margin_rate = collision_safety_margin_rate

        # mppi variables
        self.u_prev = np.zeros((self.T, self.dim_u))

        # ref_path info
        self.prev_waypoints_idx = 0

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        """calculate optimal control input (K-sample loop vectorized with numpy)"""
        # load previous control input sequence
        u = self.u_prev

        # set initial x value from observation
        x0 = observed_x

        # get the waypoint closest to current vehicle position
        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0]-1:
            print("[ERROR] Reached the end of the reference path.")
            raise IndexError

        # sample noise: shape (K, T, dim_u)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u)

        # build control input sequence with noise for all K samples at once
        # shape (K, T, dim_u)
        n_exploit = int(np.ceil((1.0 - self.param_exploration) * self.K))
        v = np.empty((self.K, self.T, self.dim_u))
        v[:n_exploit] = u[np.newaxis, :, :] + epsilon[:n_exploit]
        v[n_exploit:] = epsilon[n_exploit:]

        # initial state replicated for all K samples: (K, 4)
        x = np.tile(x0, (self.K, 1)).astype(np.float64)

        # state-cost accumulator: (K,)
        S = np.zeros((self.K,))

        # storage for sampled trajectories (used for visualization)
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))

        # time loop (K dimension is fully vectorized)
        for t in range(self.T):
            v_clamped = self._g_vec(v[:, t, :])  # (K, 2)
            x = self._F_vec(x, v_clamped)        # (K, 4)
            sampled_traj_list[:, t, :] = x

            # stage cost + control penalty
            # penalty: param_gamma * u[t].T @ Sigma_inv @ v[k, t]
            # vectorized: v[:, t, :] @ (Sigma_inv @ u[t]) -> (K,)
            penalty = v[:, t, :] @ (self.Sigma_inv @ u[t])
            S += self._c_vec(x) + self.param_gamma * penalty

        # terminal cost
        S += self._phi_vec(x)

        # compute information theoretic weights for each sample
        w = self._compute_weights(S)

        # w_epsilon[t, d] = sum_k w[k] * epsilon[k, t, d]
        w_epsilon = (w[:, np.newaxis, np.newaxis] * epsilon).sum(axis=0)

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)

        # update control input sequence
        u = u + w_epsilon

        # calculate optimal trajectory (single sample evolution, T steps only)
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualize_optimal_traj:
            x_opt = x0.copy()
            for t in range(self.T):
                x_opt = self._F(x_opt, self._g(u[t].copy()))
                optimal_traj[t] = x_opt

        # clear sampled trajectories if visualization is disabled
        if not self.visualze_sampled_trajs:
            sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))

        # update previous control input sequence (shift 1 step to the left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence
        return u[0], u, optimal_traj, sampled_traj_list

    def _calc_epsilon(self, sigma: np.ndarray, size_sample: int, size_time_step: int, size_dim_u: int) -> np.ndarray:
        """sample epsilon"""
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError
        mu = np.zeros((size_dim_u))
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step))
        return epsilon

    def _g(self, v: np.ndarray) -> np.ndarray:
        """clamp input (single sample, kept for backward compatibility)"""
        v[0] = np.clip(v[0], -self.max_steer_abs, self.max_steer_abs)
        v[1] = np.clip(v[1], -self.max_accel_abs, self.max_accel_abs)
        return v

    def _g_vec(self, v: np.ndarray) -> np.ndarray:
        """clamp input for K samples at once. v shape: (K, 2) -> (K, 2)"""
        v_out = v.copy()
        v_out[:, 0] = np.clip(v_out[:, 0], -self.max_steer_abs, self.max_steer_abs)
        v_out[:, 1] = np.clip(v_out[:, 1], -self.max_accel_abs, self.max_accel_abs)
        return v_out

    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost (single sample, kept for backward compatibility)"""
        x, y, yaw, v = x_t
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi))
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        yaw_diff = np.arctan2(np.sin(yaw-ref_yaw), np.cos(yaw-ref_yaw))
        stage_cost = (self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2
                      + self.stage_cost_weight[2]*(yaw_diff)**2 + self.stage_cost_weight[3]*(v-ref_v)**2)
        stage_cost += self._is_collided(x_t) * 1.0e10
        return stage_cost

    def _c_vec(self, x_t: np.ndarray) -> np.ndarray:
        """calculate stage cost for K samples at once. x_t shape: (K, 4) -> (K,)"""
        x = x_t[:, 0]
        y = x_t[:, 1]
        yaw = x_t[:, 2]
        v = x_t[:, 3]
        yaw = (yaw + 2.0*np.pi) % (2.0*np.pi)

        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoints_vec(x, y)
        yaw_diff = np.arctan2(np.sin(yaw-ref_yaw), np.cos(yaw-ref_yaw))
        stage_cost = (self.stage_cost_weight[0]*(x-ref_x)**2
                      + self.stage_cost_weight[1]*(y-ref_y)**2
                      + self.stage_cost_weight[2]*(yaw_diff)**2
                      + self.stage_cost_weight[3]*(v-ref_v)**2)
        stage_cost += self._is_collided_vec(x_t) * 1.0e10
        return stage_cost

    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost (single sample, kept for backward compatibility)"""
        x, y, yaw, v = x_T
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi))
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        yaw_diff = np.arctan2(np.sin(yaw-ref_yaw), np.cos(yaw-ref_yaw))
        terminal_cost = (self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2
                         + self.terminal_cost_weight[2]*(yaw_diff)**2 + self.terminal_cost_weight[3]*(v-ref_v)**2)
        terminal_cost += self._is_collided(x_T) * 1.0e10
        return terminal_cost

    def _phi_vec(self, x_T: np.ndarray) -> np.ndarray:
        """calculate terminal cost for K samples at once. x_T shape: (K, 4) -> (K,)"""
        x = x_T[:, 0]
        y = x_T[:, 1]
        yaw = x_T[:, 2]
        v = x_T[:, 3]
        yaw = (yaw + 2.0*np.pi) % (2.0*np.pi)

        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoints_vec(x, y)
        yaw_diff = np.arctan2(np.sin(yaw-ref_yaw), np.cos(yaw-ref_yaw))
        terminal_cost = (self.terminal_cost_weight[0]*(x-ref_x)**2
                         + self.terminal_cost_weight[1]*(y-ref_y)**2
                         + self.terminal_cost_weight[2]*(yaw_diff)**2
                         + self.terminal_cost_weight[3]*(v-ref_v)**2)
        terminal_cost += self._is_collided_vec(x_T) * 1.0e10
        return terminal_cost

    def _is_collided(self, x_t: np.ndarray) -> float:
        """check collision (single sample, kept for backward compatibility)"""
        vw, vl = self.vehicle_w, self.vehicle_l
        safety_margin_rate = self.collision_safety_margin_rate
        vw, vl = vw*safety_margin_rate, vl*safety_margin_rate
        x, y, yaw, _ = x_t

        vehicle_shape_x = [-0.5*vl, -0.5*vl, 0.0, +0.5*vl, +0.5*vl, +0.5*vl, 0.0, -0.5*vl, -0.5*vl]
        vehicle_shape_y = [0.0, +0.5*vw, +0.5*vw, +0.5*vw, 0.0, -0.5*vw, -0.5*vw, -0.5*vw, 0.0]
        rotated_vehicle_shape_x, rotated_vehicle_shape_y = \
            self._affine_transform(vehicle_shape_x, vehicle_shape_y, yaw, [x, y])

        for obs in self.obstacle_circles:
            obs_x, obs_y, obs_r = obs
            for p in range(len(rotated_vehicle_shape_x)):
                if (rotated_vehicle_shape_x[p]-obs_x)**2 + (rotated_vehicle_shape_y[p]-obs_y)**2 < obs_r**2:
                    return 1.0
        return 0.0

    def _is_collided_vec(self, x_t: np.ndarray) -> np.ndarray:
        """check collision for K samples at once. x_t shape: (K, 4) -> (K,)"""
        vw = self.vehicle_w * self.collision_safety_margin_rate
        vl = self.vehicle_l * self.collision_safety_margin_rate

        x = x_t[:, 0]    # (K,)
        y = x_t[:, 1]
        yaw = x_t[:, 2]

        # 9 key points around the vehicle body (in vehicle local frame)
        pts_x = np.array([-0.5*vl, -0.5*vl, 0.0, +0.5*vl, +0.5*vl, +0.5*vl, 0.0, -0.5*vl, -0.5*vl])
        pts_y = np.array([0.0, +0.5*vw, +0.5*vw, +0.5*vw, 0.0, -0.5*vw, -0.5*vw, -0.5*vw, 0.0])

        cos_yaw = np.cos(yaw)[:, np.newaxis]  # (K, 1)
        sin_yaw = np.sin(yaw)[:, np.newaxis]

        # rotated + translated points: (K, 9)
        rx = pts_x[np.newaxis, :] * cos_yaw - pts_y[np.newaxis, :] * sin_yaw + x[:, np.newaxis]
        ry = pts_x[np.newaxis, :] * sin_yaw + pts_y[np.newaxis, :] * cos_yaw + y[:, np.newaxis]

        # check each obstacle against all sample-point pairs
        collided = np.zeros(x_t.shape[0], dtype=bool)
        for obs in self.obstacle_circles:
            obs_x, obs_y, obs_r = obs
            dist_sq = (rx - obs_x)**2 + (ry - obs_y)**2  # (K, 9)
            collided |= (dist_sq < obs_r**2).any(axis=1)

        return collided.astype(np.float64)

    def _affine_transform(self, xlist: list, ylist: list, angle: float, translation: list=[0.0, 0.0]) -> Tuple[list, list]:
        transformed_x = []
        transformed_y = []
        if len(xlist) != len(ylist):
            print("[ERROR] xlist and ylist must have the same size.")
            raise AttributeError
        for i, xval in enumerate(xlist):
            transformed_x.append((xlist[i])*np.cos(angle)-(ylist[i])*np.sin(angle)+translation[0])
            transformed_y.append((xlist[i])*np.sin(angle)+(ylist[i])*np.cos(angle)+translation[1])
        transformed_x.append(transformed_x[0])
        transformed_y.append(transformed_y[0])
        return transformed_x, transformed_y

    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle (single sample, kept for backward compatibility)"""
        SEARCH_IDX_LEN = 200
        prev_idx = self.prev_waypoints_idx
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx
        ref_x = self.ref_path[nearest_idx, 0]
        ref_y = self.ref_path[nearest_idx, 1]
        ref_yaw = self.ref_path[nearest_idx, 2]
        ref_v = self.ref_path[nearest_idx, 3]
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx
        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v

    def _get_nearest_waypoints_vec(self, x: np.ndarray, y: np.ndarray):
        """find nearest waypoint for K samples at once. x, y shape (K,) -> indices and ref values, all (K,)"""
        SEARCH_IDX_LEN = 200
        prev_idx = self.prev_waypoints_idx
        search_slice = self.ref_path[prev_idx:prev_idx + SEARCH_IDX_LEN]  # (n, 4) where n<=200

        # distance squared from each sample to each search waypoint: (K, n)
        dx = x[:, np.newaxis] - search_slice[np.newaxis, :, 0]
        dy = y[:, np.newaxis] - search_slice[np.newaxis, :, 1]
        d_sq = dx**2 + dy**2

        rel_idx = np.argmin(d_sq, axis=1)            # (K,)
        nearest_idx = rel_idx + prev_idx              # (K,)
        ref_x = search_slice[rel_idx, 0]
        ref_y = search_slice[rel_idx, 1]
        ref_yaw = search_slice[rel_idx, 2]
        ref_v = search_slice[rel_idx, 3]
        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v

    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state (single sample, kept for backward compatibility)"""
        x, y, yaw, v = x_t
        steer, accel = v_t
        l = self.wheel_base
        dt = self.delta_t
        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + v / l * np.tan(steer) * dt
        new_v = v + accel * dt
        return np.array([new_x, new_y, new_yaw, new_v])

    def _F_vec(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state for K samples at once. x_t: (K, 4), v_t: (K, 2) -> (K, 4)"""
        x = x_t[:, 0]
        y = x_t[:, 1]
        yaw = x_t[:, 2]
        v = x_t[:, 3]
        steer = v_t[:, 0]
        accel = v_t[:, 1]
        l = self.wheel_base
        dt = self.delta_t

        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + v / l * np.tan(steer) * dt
        new_v = v + accel * dt

        return np.stack([new_x, new_y, new_yaw, new_v], axis=1)

    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample (vectorized)"""
        rho = S.min()
        exp_terms = np.exp((-1.0/self.param_lambda) * (S - rho))
        return exp_terms / exp_terms.sum()

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence"""
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:, d] = np.convolve(xx[:, d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0, d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i, d] *= window_size/(i+n_conv)
                xx_mean[-i, d] *= window_size/(i + n_conv - (window_size % 2))
        return xx_mean


def run_simulation_mppi_pathtracking() -> None:
    """run simulation of pathtracking with MPPI controller"""
    print("[INFO] Start simulation of pathtracking with MPPI controller")

    # simulation settings
    delta_t = 0.05 # [sec]
    sim_steps = 150 # [steps]
    print(f"[INFO] delta_t : {delta_t:.2f}[s] , sim_steps : {sim_steps}[steps], total_sim_time : {delta_t*sim_steps:.2f}[s]")

    # obstacle params
    OBSTACLE_CIRCLES = np.array([
        [+ 8.0, +5.0, 4.0],
        [+18.0, -5.0, 4.0],
    ])

    # load and visualize reference path
    ref_path = np.genfromtxt('./data/ovalpath.csv', delimiter=',', skip_header=1)

    # initialize a vehicle as a control target
    vehicle = Vehicle(
        wheel_base=2.5,
        max_steer_abs=0.523,
        max_accel_abs=2.000,
        ref_path = ref_path[:, 0:2],
        obstacle_circles = OBSTACLE_CIRCLES,
    )
    vehicle.reset(
        init_state = np.array([0.0, 0.0, 0.0, 0.0]),
    )

    # initialize a mppi controller for the vehicle
    mppi = MPPIControllerForPathTracking(
        delta_t = delta_t*2.0,
        wheel_base = 2.5,
        max_steer_abs = 0.523,
        max_accel_abs = 2.000,
        ref_path = ref_path,
        horizon_step_T = 20,
        number_of_samples_K = 500,
        param_exploration = 0.05,
        param_lambda = 100.0,
        param_alpha = 0.98,
        sigma = np.array([[0.075, 0.0], [0.0, 2.0]]),
        stage_cost_weight = np.array([50.0, 50.0, 1.0, 20.0]),
        terminal_cost_weight = np.array([50.0, 50.0, 1.0, 20.0]),
        visualze_sampled_trajs = True,
        obstacle_circles = OBSTACLE_CIRCLES,
        collision_safety_margin_rate = 1.2,
    )

    # simulation loop
    for i in range(sim_steps):

        current_state = vehicle.get_state()

        try:
            optimal_input, optimal_input_sequence, optimal_traj, sampled_traj_list = mppi.calc_control_input(
                observed_x = current_state
            )
        except IndexError as e:
            print("[ERROR] IndexError detected. Terminate simulation.")
            break

        print(f"Time: {i*delta_t:>2.2f}[s], x={current_state[0]:>+3.3f}[m], y={current_state[1]:>+3.3f}[m], yaw={current_state[2]:>+3.3f}[rad], v={current_state[3]:>+3.3f}[m/s], steer={optimal_input[0]:>+6.2f}[rad], accel={optimal_input[1]:>+6.2f}[m/s]")

        vehicle.update(u=optimal_input, delta_t=delta_t, optimal_traj=optimal_traj[:, 0:2], sampled_traj_list=sampled_traj_list[:, :, 0:2])

    vehicle.save_animation("mppi_pathtracking_obav_demo.mp4", interval=int(delta_t * 1000), movie_writer="ffmpeg")

if __name__ == "__main__":
    run_simulation_mppi_pathtracking()