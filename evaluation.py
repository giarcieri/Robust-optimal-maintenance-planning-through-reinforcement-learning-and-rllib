import argparse
from hmm_AR_k_Tstud import HMMStates, TruncatedNormalEmissionsAR_k
from env import *
import pickle
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

parser = argparse.ArgumentParser(description='rlfr-rllib')
parser.add_argument('-m', '--model', type=str, metavar='',
                    required=True, help='lstm, gtrxl or belief')
parser.add_argument('-i', '--iterations', type=int, metavar='',
                    required=True)
args = parser.parse_args()

model = str(args.model)
iterations = int(args.iterations)

if model not in ['lstm', 'gtrxl', 'belief', 'belief_dr']:
    raise ValueError(f'model is not lstm, gtrxl, or belief but {model}')


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
register_env("fractal_env", env_creator)

if model == 'belief':
    algo = Algorithm.from_checkpoint('./checkpoints_belief_DR_False/checkpoint_001960')
elif model == 'gtrxl':
    algo = Algorithm.from_checkpoint('./checkpoints_gtrxl_DR_False/checkpoint_008190')
elif model == 'lstm':
    algo = Algorithm.from_checkpoint('./checkpoints_lstm_DR_False/checkpoint_019815')
elif model == 'belief_dr':
    algo = Algorithm.from_checkpoint('./checkpoints_belief_DR_True/checkpoint_004715')

print(f'Evaluate {model} on {iterations} iterations')
rewards_list = []
for i in range(iterations//500):
    results = algo.evaluate()
    #print(results)
    rewards = results['evaluation']['hist_stats']['episode_reward']
    rewards_list.append(rewards)
rewards_list = np.array(rewards_list).reshape(-1,)
print(rewards_list.mean(), rewards_list.max(), rewards_list.min(), rewards_list.std(), rewards_list.std()/(iterations)**0.5, rewards_list.shape)
np.save(f'eval_rewards_{model}.npy', rewards_list)
