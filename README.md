# Robust-optimal-maintenance-planning-through-reinforcement-learning-and-rllib

Code of the paper "POMDP inference and robust solution via deep reinforcement learning: An application to railway optimal maintenance", currently under review. 

## Requirements

```
conda create -n rllib python=3.8.13
conda activate rllib
pip install -r requirements.txt
```

## Running

The environment is implemented in ``env.py``. You can modify the hyperparameters of the model and the PPO algorithm in ``config_*.json`` and run the training with the command:

```
export OMP_NUM_THREADS=50; python main.py --model $MODEL
```
with ``$MODEL`` in [belief, gtrxl, lstm]. Alternatively, you can submit the job via bsub command:

```
bsub < job.bsub
```

The training will save the average rewards at every evaluation iteration and the best model. You can then run a longer evaluation of the best model by submitting ``eval.bsub`` or running ``evaluation.py``.
