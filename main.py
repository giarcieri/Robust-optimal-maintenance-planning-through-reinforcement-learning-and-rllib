import pprint
import json
import ray
import pickle
import argparse
import numpy as np
from ray import tune
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG
from ray.tune.registry import register_env
from env import FractalEnv
from hmm_AR_k_Tstud import HMMStates, TruncatedNormalEmissionsAR_k

parser = argparse.ArgumentParser(description='rlfr-rllib')
parser.add_argument('-m', '--model', type=str, metavar='',
                    required=True, help='lstm, gtrxl or belief')
args = parser.parse_args()

model = str(args.model)

if model not in ['lstm', 'gtrxl', 'belief']:
    raise ValueError(f'model is not lstm or gtrxl but {model}')

trace_file = 'trace.pickle'
with open(trace_file, "rb") as fp:
    trace = pickle.load(fp)

reward_a_0 = - 0
reward_a_R2 = - 50
reward_a_A1 = - 2000 


reward_s_0 = - 100
reward_s_1 = - 200
reward_s_2 = - 1000
reward_s_3 = - 8000 #10000

reward_matrix = np.asarray([
    [reward_a_0 + reward_s_0, reward_a_0 + reward_s_1, reward_a_0 + reward_s_2, reward_a_0 + reward_s_3],
    [reward_a_R2 + reward_s_0, reward_a_R2 + reward_s_1, reward_a_R2 + reward_s_2, reward_a_R2 + reward_s_3],
    [1*reward_a_A1 + reward_a_R2 + reward_s_0, 1.33*reward_a_A1 + reward_a_R2 + reward_s_1, 1.66*reward_a_A1 + reward_a_R2 + reward_s_2, 2*reward_a_A1 + reward_a_R2 + reward_s_3]
])


def env_creator(env_config):
    return FractalEnv(trace=trace, reward_matrix=reward_matrix, env_config=env_config)

with open(f'config_{model}.json') as file:
    config_file = json.load(file)

def run_main(config_params=config_file):
    config = DEFAULT_CONFIG.copy()
    pp = pprint.PrettyPrinter(indent=4)
    register_env("fractal_env", env_creator)
    config['env'] = "fractal_env"
    config.update(config_params)
    pp.pprint(config)
    ray.init()
    trainer = PPOTrainer(config=config)
    for iteration in range(30000):
        results = trainer.train()
        try:
            mean_rewards = results['evaluation']['episode_reward_mean']
            with open(f"results_{model}.txt", "a") as f:
                f.write(f"{mean_rewards}\n")
        except:
            pass
    pp.pprint(results)
    #checkpoint_dir = trainer.save(checkpoint_dir="./checkpoints")
    #print(f"Checkpoint saved in directory {checkpoint_dir}")


if __name__ == '__main__':
    run_main(config_file)