import numpy as np
import gym
from typing import Tuple, Union, Dict
from ray.rllib.env.env_context import EnvContext
import pymc3 as pm
from numpyro.distributions import StudentT, TruncatedDistribution

class FractalEnv(gym.Env):
    """
    Numpy implementation of the fractal environment.
    """

    def __init__(
        self,
        trace: Dict,
        reward_matrix: np.ndarray,
        #seed: int,
        env_config: EnvContext
    ) -> None:

        self.trace = trace
        self.reward_matrix = reward_matrix
        self.domain_randomization = env_config['domain_randomization']
        self.print_variables = env_config['print_variables'] 
        self.return_belief = env_config['return_belief'] 
        self.worker_index = env_config.worker_index
        self.NegativeStudentT = pm.Bound(pm.StudentT, upper=0.0).dist
        #np.random.seed(seed=seed)
        self.action_space = gym.spaces.Discrete(3)
        if self.return_belief:
            self.observation_space = gym.spaces.Box(low=np.zeros(4), high=np.ones(4), shape=(4,), dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(low=np.NINF, high=np.array([0.]), shape=(1,), dtype=np.float32)

    def reset(
        self, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        # reset params
        if self.domain_randomization:
            self.sample_params()
        else:
            self.sample_mean_params()

        # sample initial state
        init_probs = self.params['init_probs']
        self.state = np.random.choice(np.arange(4), p=init_probs)
        self.belief = init_probs

        # sample initial obs
        self.obs = self.init_process(self.state)

        self.t = 0

        if self.return_belief:
            return self.belief
        else:
            return self.obs

    def step(
        self,
        action: Union[int, float],
    ) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:

        # sample reward
        reward = self.reward_matrix[action, self.state]

        self.t += 1

        # print variables
        if self.print_variables and self.worker_index == 1:
            with open("variables.txt", "a") as f:
                f.write(f'Timestep {self.t} obs {self.obs}, state {self.state}, action {action}, reward {reward}\n')

        # sample new state
        transition_matrices = self.params['p_transition']
        self.state = np.random.choice(np.arange(4), p=transition_matrices[action, self.state].squeeze())

        # sample obs
        if action == 0:
            obs = self.deterioration_process(self.state, self.obs)
        else:
            obs = self.repair_process(self.state, self.obs, action)

        if self.return_belief:
            self.update_belief(real_observation=obs, real_action=action, prev_obs=self.obs)
            self.obs = obs
            return self.belief, reward, False, {'state': self.state, 'obs': self.obs}
        else:
            self.obs = obs
            return self.obs, reward, False, {'state': self.state}

    def sample_params(self): 
        trace = self.trace
        n_samples = trace['p_transition'].shape[0]
        index_sample = np.random.choice(a=n_samples)

        transition_matrices = trace['p_transition'][index_sample]
        init_probs = trace['init_probs'][index_sample]

        mu_d = trace['mu_d'][index_sample]
        sigma_d = trace['sigma_d'][index_sample]
        nu_d = trace['nu_d'][index_sample]

        mu_r = trace['mu_r'][index_sample]
        sigma_r = trace['sigma_r'][index_sample]
        nu_r = trace['nu_r'][index_sample]

        mu_init = trace['mu_init'][index_sample]
        sigma_init = trace['sigma_init'][index_sample]
        nu_init = trace['nu_init'][index_sample]

        k = trace['k'][index_sample]

        self.params = {
            'init_probs': init_probs,
            'p_transition': transition_matrices, 
            'mu_d': mu_d,
            'sigma_d': sigma_d, 
            'nu_d': nu_d,
            'mu_r': mu_r,
            'sigma_r': sigma_r,
            'nu_r': nu_r,
            'mu_init': mu_init,
            'sigma_init': sigma_init,
            'nu_init': nu_init,
            'k': k
        }

    def sample_mean_params(self): 
        trace = self.trace
        transition_matrices = trace['p_transition'].mean(0)
        init_probs = trace['init_probs'].mean(0)

        mu_d = trace['mu_d'].mean(0)
        sigma_d = trace['sigma_d'].mean(0)
        nu_d = trace['nu_d'].mean(0)

        mu_r = trace['mu_r'].mean(0)
        sigma_r = trace['sigma_r'].mean(0)
        nu_r = trace['nu_r'].mean(0)

        mu_init = trace['mu_init'].mean(0)
        sigma_init = trace['sigma_init'].mean(0)
        nu_init = trace['nu_init'].mean(0)

        k = trace['k'].mean(0)
        self.params = {
            'init_probs': init_probs,
            'p_transition': transition_matrices, 
            'mu_d': mu_d,
            'sigma_d': sigma_d, 
            'nu_d': nu_d,
            'mu_r': mu_r,
            'sigma_r': sigma_r,
            'nu_r': nu_r,
            'mu_init': mu_init,
            'sigma_init': sigma_init,
            'nu_init': nu_init,
            'k': k
        }

    @property
    def name(self) -> str:
        """Environment name."""
        return "Fractal-values"

    def deterioration_process(self, state, obs):
        params = self.params
        mu_d, sigma_d, nu_d = params['mu_d'][state], params[
            'sigma_d'][state], params['nu_d'][state]
        DetStudentT = pm.Bound(pm.StudentT, upper=float(-obs)).dist
        sample = DetStudentT(mu=mu_d, sigma=sigma_d, nu=nu_d).random()
        return sample + obs

    def repair_process(self, state, obs, action):
        params = self.params
        mu_r, sigma_r, nu_r, k = params['mu_r'][state], params['sigma_r'][
            state], params['nu_r'][state], params['k'][action-1]
        sample = self.NegativeStudentT(mu=k*obs + mu_r, sigma=sigma_r, nu=nu_r).random()
        return sample

    def init_process(self, state):
        params = self.params
        mu_init, sigma_init, nu_init = params['mu_init'][state], params['sigma_init'][state], params['nu_init'][state]
        sample = self.NegativeStudentT(mu=mu_init, sigma=sigma_init, nu=nu_init).random()
        return sample

    def observation_model_probability(self, observation, next_state, action, prev_obs):
        params = self.params
        _low = -100.
        if action == 0:
            mu_d, sigma_d, nu_d = params['mu_d'][next_state], params['sigma_d'][next_state], params['nu_d'][next_state]
            detstud = TruncatedDistribution(StudentT(df=nu_d, loc=mu_d, scale=sigma_d), low=_low, high=-prev_obs)
            return np.exp(detstud.log_prob(observation-prev_obs))
        elif action in [1, 2]:
            mu_r, sigma_r, nu_r, k = params['mu_r'][next_state], params['sigma_r'][next_state], params['nu_r'][next_state], params['k'][action-1]
            negstud = TruncatedDistribution(StudentT(df=nu_r, loc=k*prev_obs + mu_r, scale=sigma_r), low=_low, high=0.)
            return np.exp(negstud.log_prob(observation))
        else:
            raise ValueError(f'Invalid input: s_t {next_state}, a_t {action}, o_t-1 {prev_obs}')

    def update_belief(self, real_observation, real_action, prev_obs):
        transition_matrices = self.params['p_transition']
        new_belief = np.zeros(self.belief.shape)
        total_prob = 0
        for next_state in np.arange(4):
            observation_prob = self.observation_model_probability(real_observation,
                                                                  next_state,
                                                                  real_action,
                                                                  prev_obs)
            transition_prob = 0
            for state in np.arange(4):
                transition_prob += transition_matrices[real_action, state, next_state] * self.belief[state]
            
            new_belief[next_state] = observation_prob * transition_prob
            total_prob += new_belief[next_state]
        new_belief /= total_prob
        self.belief = new_belief
