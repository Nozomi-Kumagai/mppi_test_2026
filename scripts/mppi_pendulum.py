import math
import numpy as np
from pendulum import Pendulum
from typing import Tuple

class MPPIControllerForPendulum():
    def __init__(
            self,
            delta_t: float = 0.05,
            mass_of_pole: float = 1.0,
            length_of_pole: float = 1.0,
            max_torque_abs: float = 2.0,
            max_speed_abs: float = 8.0,
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.01,
            param_lambda: float = 1.0,
            param_alpha: float = 0.1,
            sigma: float = 1.0,
            stage_cost_weight: np.ndarray = np.array([1.0, 0.1]),
            terminal_cost_weight: np.ndarray = np.array([1.0, 0.1]),
    ) -> None:
        """initialize mppi controller for pendulum"""
        # mppi parameters
        self.dim_u = 1 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration
        self.param_lambda = param_lambda
        self.param_alpha = param_alpha
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))
        self.Sigma = sigma
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight

        # pendulum parameters
        self.g = 9.81
        self.delta_t = delta_t
        self.mass_of_pole = mass_of_pole
        self.length_of_pole = length_of_pole
        self.max_torque = max_torque_abs
        self.max_speed = max_speed_abs

        # mppi variables
        self.u_prev = np.zeros((self.T))

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        """calculate optimal control input (K-sample loop vectorized with numpy)"""
        # load previous control input sequence
        u = self.u_prev

        # set initial x value from observation
        x0 = observed_x

        # sample noise: shape (K, T)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T)

        # build control input sequence with noise for all K samples at once: (K, T)
        n_exploit = int((1.0 - self.param_exploration) * self.K)
        v = np.empty((self.K, self.T))
        v[:n_exploit, :] = u[np.newaxis, :] + epsilon[:n_exploit, :]
        v[n_exploit:, :] = epsilon[n_exploit:, :]

        # clamp v with max_torque
        v_clamped = np.clip(v, -self.max_torque, self.max_torque)

        # initial state replicated for all K samples: shape (K, 2)
        x = np.tile(x0, (self.K, 1)).astype(np.float64)

        # state-cost accumulator: shape (K,)
        S = np.zeros((self.K,))

        # time loop (K dimension is fully vectorized)
        for t in range(self.T):
            # propagate all K samples one step
            x = self._F_vec(x, v_clamped[:, t])

            # stage cost: per-sample cost + control penalty term
            S += self._c_vec(x) + self.param_gamma * u[t] * (1.0 / self.Sigma) * v[:, t]

        # terminal cost (vectorized)
        S += self._phi_vec(x)

        # compute information theoretic weights for each sample
        w = self._compute_weights(S)

        # w_epsilon[t] = sum_k w[k] * epsilon[k, t]
        w_epsilon = (w[:, np.newaxis] * epsilon).sum(axis=0)

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=5)

        # update control input sequence
        u = u + w_epsilon

        # update previous control input sequence (shift 1 step to the left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence
        return u[0], u

    def _calc_epsilon(self, sigma: float, size_sample: int, size_time_step: int) -> np.ndarray:
        """sample epsilon"""
        epsilon = np.random.normal(0.0, sigma, (self.K, self.T))
        return epsilon

    def _g(self, v: np.ndarray) -> float:
        """clamp input (single sample, kept for backward compatibility)"""
        v = np.clip(v, -self.max_torque, self.max_torque)
        return v

    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost (single sample, kept for backward compatibility)"""
        theta, theta_dot = x_t[0], x_t[1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        return self.stage_cost_weight[0]*theta**2 + self.stage_cost_weight[1]*theta_dot**2

    def _c_vec(self, x_t: np.ndarray) -> np.ndarray:
        """stage cost for K samples at once. x_t shape: (K, 2) -> (K,)"""
        theta = x_t[:, 0]
        theta_dot = x_t[:, 1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        return self.stage_cost_weight[0] * theta**2 + self.stage_cost_weight[1] * theta_dot**2

    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost (single sample, kept for backward compatibility)"""
        theta, theta_dot = x_T[0], x_T[1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        return self.terminal_cost_weight[0]*theta**2 + self.terminal_cost_weight[1]*theta_dot**2

    def _phi_vec(self, x_T: np.ndarray) -> np.ndarray:
        """terminal cost for K samples at once. x_T shape: (K, 2) -> (K,)"""
        theta = x_T[:, 0]
        theta_dot = x_T[:, 1]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        return self.terminal_cost_weight[0] * theta**2 + self.terminal_cost_weight[1] * theta_dot**2

    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state (single sample, kept for backward compatibility)"""
        theta, theta_dot = x_t[0], x_t[1]
        g = self.g
        m = self.mass_of_pole
        l = self.length_of_pole
        dt = self.delta_t
        torque = v_t
        new_theta_dot = theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * torque) * dt
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * dt
        return np.array([new_theta, new_theta_dot])

    def _F_vec(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """next state for K samples at once. x_t: (K, 2), v_t: (K,) -> (K, 2)"""
        theta = x_t[:, 0]
        theta_dot = x_t[:, 1]
        g = self.g
        m = self.mass_of_pole
        l = self.length_of_pole
        dt = self.delta_t
        torque = v_t

        new_theta_dot = theta_dot + (3 * g / (2 * l) * np.sin(theta) + 3.0 / (m * l**2) * torque) * dt
        new_theta_dot = np.clip(new_theta_dot, -self.max_speed, self.max_speed)
        new_theta = theta + new_theta_dot * dt

        return np.stack([new_theta, new_theta_dot], axis=1)

    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample (vectorized)"""
        rho = S.min()
        exp_terms = np.exp((-1.0 / self.param_lambda) * (S - rho))
        return exp_terms / exp_terms.sum()

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        Note: The original MPPI paper uses the Savitzky-Golay Filter for smoothing control inputs.
        """
        b = np.ones(window_size)/window_size
        xx_mean = np.convolve(xx, b, mode="same")
        n_conv = math.ceil(window_size/2)
        xx_mean[0] *= window_size/n_conv
        for i in range(1, n_conv):
            xx_mean[i] *= window_size/(i+n_conv)
            xx_mean[-i] *= window_size/(i + n_conv - (window_size % 2))
        return xx_mean


def run_simulation_mppi_pendulum() -> None:
    """run simulation of swinging up pendulum with MPPI controller"""
    print("[INFO] Start simulation of swinging up a pendulum with MPPI controller")

    # simulation settings
    delta_t = 0.05 # [sec]
    sim_steps = 150 # [steps]
    print(f"[INFO] delta_t : {delta_t:.2f}[s] , sim_steps : {sim_steps}[steps], total_sim_time : {delta_t*sim_steps:.2f}[s]")

    # initialize a pendulum as a control target
    pendulum = Pendulum(
        mass_of_pole = 1.0,
        length_of_pole = 1.0,
        max_torque_abs = 2.0,
        max_speed_abs = 8.0,
        delta_t = delta_t,
        visualize = True,
    )
    pendulum.reset(
        init_state = np.array([np.pi, 0.0]),
    )

    # initialize a mppi controller for the pendulum
    mppi = MPPIControllerForPendulum(
        delta_t = delta_t,
        mass_of_pole = 1.0,
        length_of_pole = 1.0,
        max_torque_abs = 2.0,
        max_speed_abs = 8.0,
        horizon_step_T = 20,
        number_of_samples_K = 2000,
        param_exploration = 0.05,
        param_lambda = 0.5,
        param_alpha = 0.8,
        sigma = 1.0,
        stage_cost_weight    = np.array([1.0, 0.1]),
        terminal_cost_weight = 5.0 * np.array([1.0, 0.1]),
    )

    # simulation loop
    for i in range(sim_steps):

        current_state = pendulum.get_state()

        input_torque, input_torque_sequence = mppi.calc_control_input(
            observed_x = current_state
        )

        print(f"Time: {i*delta_t:>2.2f}[s], theta={current_state[0]:>+3.3f}[rad], theta_dot={current_state[1]:>+3.3f}[rad/s], input torque={input_torque:>+3.2f}[Nm]", end="")
        print(", # currently staying upright #" if abs(current_state[0]) < 0.1 and abs(current_state[1] < 0.1) else "")

        pendulum.update(u=[input_torque], delta_t=delta_t)

    pendulum.save_animation("mppi_pendulum.mp4", interval=int(delta_t * 1000), movie_writer="ffmpeg")


if __name__ == "__main__":
    run_simulation_mppi_pendulum()