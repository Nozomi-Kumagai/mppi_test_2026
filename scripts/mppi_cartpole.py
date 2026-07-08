import math
import numpy as np
from cartpole import CartPole
from typing import Tuple

class MPPIControllerForCartPole():
    def __init__(
            self,
            delta_t: float = 0.02,
            mass_of_cart: float = 1.0,
            mass_of_pole: float = 0.01,
            length_of_pole: float = 2.0,
            max_force_abs: float = 100.0,
            horizon_step_T: int = 100,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: float = 10.0,
            stage_cost_weight: np.ndarray = np.array([5.0, 10.0, 0.1, 0.1]), # weight for [x, theta, x_dot, theta_dot]
            terminal_cost_weight: np.ndarray = np.array([5.0, 10.0, 0.1, 0.1]) # weight for [x, theta, x_dot, theta_dot]
    ) -> None:
        """initialize mppi controller for cartpole"""
        # mppi parameters
        self.dim_u = 1 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi
        self.Sigma = sigma # deviation of noise
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight

        # cartpole parameters
        self.g = 9.81
        self.delta_t = delta_t
        self.mass_of_cart = mass_of_cart
        self.mass_of_pole = mass_of_pole
        self.length_of_pole = length_of_pole
        self.max_force_abs = max_force_abs

        # mppi variables
        self.u_prev = np.zeros((self.T))

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        u = self.u_prev
        x0 = observed_x

        # sample noise: (K, T)
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T)

        # build v for all K samples at once: (K, T)
        n_exploit = int(np.ceil((1.0 - self.param_exploration) * self.K))
        v = np.empty((self.K, self.T))
        v[:n_exploit, :] = u[np.newaxis, :] + epsilon[:n_exploit, :]
        v[n_exploit:, :] = epsilon[n_exploit:, :]
        v_clamped = np.clip(v, -self.max_force_abs, self.max_force_abs)

        # initial state for K samples: (K, 4)
        x = np.tile(x0, (self.K, 1)).astype(np.float64)
        S = np.zeros((self.K,))

        # time loop only (K dimension is vectorized)
        for t in range(self.T):
            x = self._F_vec(x, v_clamped[:, t])
            S += self._c_vec(x) + self.param_gamma * u[t] * (1.0/self.Sigma) * v[:, t]
        S += self._phi_vec(x)

        # weights and update
        w = self._compute_weights(S)
        w_epsilon = (w[:, np.newaxis] * epsilon).sum(axis=0)
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)
        u = u + w_epsilon

        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]
        return u[0], u
    def _calc_epsilon(self, sigma: float, size_sample: int, size_time_step: int) -> np.ndarray:
        """sample epsilon"""
        epsilon = np.random.normal(0.0, sigma, (self.K, self.T)) # size is self.K x self.T
        return epsilon

    def _g(self, v: np.ndarray) -> float:
        """clamp input"""
        v = np.clip(v, -self.max_force_abs, self.max_force_abs)
        return v

    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost"""
        # parse x_t
        x, x_dot = x_t[0], x_t[2]
        theta, theta_dot = x_t[1], x_t[3]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # calculate stage cost # (np.cos(theta)+1.0)
        stage_cost = self.stage_cost_weight[0]*x**2 + self.stage_cost_weight[1]*theta**2 + self.stage_cost_weight[2]*x_dot**2 + self.stage_cost_weight[3]*theta_dot**2
        return stage_cost
    
    def _c_vec(self, x_t):
        """vectorized version of _c. x_t: (K,4) -> (K,)"""
        x, theta, x_dot, theta_dot = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        return (self.stage_cost_weight[0]*x**2 + self.stage_cost_weight[1]*theta**2
                + self.stage_cost_weight[2]*x_dot**2 + self.stage_cost_weight[3]*theta_dot**2)
    
    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost"""
        # parse x_T
        x, x_dot = x_T[0], x_T[2]
        theta, theta_dot = x_T[1], x_T[3]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi # normalize theta to [-pi, pi]

        # calculate terminal cost # (np.cos(theta)+1.0)
        terminal_cost = self.terminal_cost_weight[0]*x**2 + self.terminal_cost_weight[1]*theta**2 + self.terminal_cost_weight[2]*x_dot**2 + self.terminal_cost_weight[3]*theta_dot**2
        return terminal_cost
    
    def _phi_vec(self, x_T):
        """vectorized version of _phi. x_T: (K,4) -> (K,)"""
        x, theta, x_dot, theta_dot = x_T[:, 0], x_T[:, 1], x_T[:, 2], x_T[:, 3]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        return (self.terminal_cost_weight[0]*x**2 + self.terminal_cost_weight[1]*theta**2
                + self.terminal_cost_weight[2]*x_dot**2 + self.terminal_cost_weight[3]*theta_dot**2)
    
    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state of the cartpole"""
        # get previous state variables
        x, theta, x_dot, theta_dot = x_t[0], x_t[1], x_t[2], x_t[3]

        # prepare params
        g = self.g
        M = self.mass_of_cart
        m = self.mass_of_pole
        l = self.length_of_pole
        f = v_t
        dt = self.delta_t

        # get acc. values
        temp = (
            f - (m*l) * theta_dot**2 * np.sin(theta)
        ) / (M + m)
        new_theta_ddot = (g * np.sin(theta) + np.cos(theta) * temp) / (
            l * (4.0 / 3.0 - m * np.cos(theta)**2 / (M + m))
        )
        new_x_ddot = temp + (m*l)  * new_theta_ddot * np.cos(theta) / (M+m)

        # update pos. values
        theta = theta + theta_dot * dt
        x = x + x_dot * dt

        # update vel. values
        theta_dot = theta_dot + new_theta_ddot * dt
        x_dot = x_dot + new_x_ddot * dt

        # return updated state
        x_t_plus_1 = np.array([x, theta, x_dot, theta_dot])
        return x_t_plus_1
    
    def _F_vec(self, x_t, v_t):
        """vectorized version of _F. x_t: (K,4), v_t: (K,) -> (K,4)"""
        x, theta, x_dot, theta_dot = x_t[:, 0], x_t[:, 1], x_t[:, 2], x_t[:, 3]
        g, M, m, l, dt = self.g, self.mass_of_cart, self.mass_of_pole, self.length_of_pole, self.delta_t
        f = v_t
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        temp = (f - (m*l) * theta_dot**2 * sin_t) / (M + m)
        new_theta_ddot = (g * sin_t + cos_t * temp) / (l * (4.0/3.0 - m * cos_t**2 / (M + m)))
        new_x_ddot = temp + (m*l) * new_theta_ddot * cos_t / (M+m)
        return np.stack([x + x_dot*dt, theta + theta_dot*dt,
                         x_dot + new_x_ddot*dt, theta_dot + new_theta_ddot*dt], axis=1)


    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""
        # calculate rho
        rho = S.min()
        exp_terms = np.exp((-1.0/self.param_lambda) * (S - rho))
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


def run_simulation_mppi_cartpole() -> None:
    """run simulation of cartpole with MPPI controller"""
    print("[INFO] Start simulation of cartpole with MPPI controller")

    # simulation settings
    delta_t = 0.02 # [sec]
    sim_steps = 200 # [steps]
    print(f"[INFO] delta_t : {delta_t:.2f}[s] , sim_steps : {sim_steps}[steps], total_sim_time : {delta_t*sim_steps:.2f}[s]")

    # initialize a cartpole as a control target
    cartpole = CartPole(
        mass_of_cart = 1.0,
        mass_of_pole = 0.01,
        length_of_pole = 2.0, 
        max_force_abs = 100.0,
    )
    cartpole.reset(
        init_state = np.array([0.0, np.pi, 0.0, 0.0]), # [x[m], theta[rad], x_dot[m/s], theta_dot[rad/s]]
    )

    # initialize a mppi controller for the cartpole
    mppi = MPPIControllerForCartPole(
        delta_t = delta_t,
        mass_of_cart = 1.0,
        mass_of_pole = 0.01,
        length_of_pole = 2.0,
        max_force_abs = 100.0,
        horizon_step_T = 100,
        number_of_samples_K = 1000,
        param_exploration = 0.0,
        param_lambda = 50.0,
        param_alpha = 1.0,
        sigma = 10.0,
        stage_cost_weight    = np.array([5.0, 10.0, 0.1, 0.1]), # weight for [x, theta, x_dot, theta_dot]
        terminal_cost_weight = np.array([5.0, 10.0, 0.1, 0.1]), # weight for [x, theta, x_dot, theta_dot]
    )

    # simulation loop
    for i in range(sim_steps):

        # get current state of cartpole
        current_state = cartpole.get_state()

        # calculate input force with MPPI
        input_force, input_force_sequence = mppi.calc_control_input(
            observed_x = current_state
        )

        # print current state and input force
        print(f"Time: {i*delta_t:>2.2f}[s], x={current_state[0]:>+3.3f}[m], theta={current_state[1]:>+3.3f}[rad], input force={input_force:>+6.2f}[N]")

        # update states of cartpole
        cartpole.update(u=[input_force], delta_t=delta_t)

    # save animation
    cartpole.save_animation("mppi_cartpole.mp4", interval=int(delta_t * 1000), movie_writer="ffmpeg") # ffmpeg is required to write mp4 file

if __name__ == "__main__":
    run_simulation_mppi_cartpole()
