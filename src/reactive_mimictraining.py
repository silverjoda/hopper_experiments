from policy import Policy
from utils import Logger, Scaler
import time

from train import init_gym, run_episode
from networks import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def evaluate_mimic(mimicpolicy, masterpolicy, scaler, env, n):
    total_mse = 0
    for i in range(n):
        observes, actions, _, _ = run_episode(env,
                                              masterpolicy,
                                              scaler,
                                              animate=False)
        total_mse += mimicpolicy.evaluate(observes, actions)

    return total_mse/n

def train_mimic(mimicpolicy,
                masterpolicy,
                env,
                scaler,
                n_episodes=100,
                batchsize=32):

    for i in range(n_episodes):
        observes, actions, _, _ = run_episode(env,
                                            masterpolicy,
                                            scaler,
                                            animate=False)

        mse = mimicpolicy.train(observes, actions, batchsize)

        print("Episode {}/{}, mse: {}".format(i, n_episodes, mse))



def main():

    RESTORE_MIMIC = False
    TRAIN = False
    EVALUATE_MIMIC = False
    VISUALIZE_MIMIC = True
    VISUALIZE_MASTER = False

    # Open AI gym environment
    env, obs_dim, act_dim = init_gym('Hopper-v1')

    # Make master policy which was trained by R
    masterpolicy = Policy(obs_dim, act_dim, 0.003, stochastic_policy=False)
    masterpolicy.restore_weights()
    scaler = Scaler(obs_dim)

    if VISUALIZE_MASTER:
        masterpolicy.visualize(env)

    print("Loaded masterpolicy. ")

    mimicpolicy = ReactivePolicy(obs_dim, act_dim)

    if RESTORE_MIMIC:
        mimicpolicy.restore_weights()

    if TRAIN:
        # Train the mimic by master
        t1 = time.time()
        train_mimic(mimicpolicy,
                    masterpolicy,
                    env,
                    scaler,
                    n_episodes=25,
                    batchsize=32)
        print("Training time taken: {}".format(time.time() - t1))
        mimicpolicy.save_weights()
        print("Saved mimic weights")


    if EVALUATE_MIMIC:
        # Evaluate the policy by MSE
        print("Starting evaluation...")
        mse = evaluate_mimic(mimicpolicy, masterpolicy, scaler, env, 10)
        print("Average MSE of mimic: {}".format(mse))

    if VISUALIZE_MIMIC:
        # Visualise the policy
        print("Visualizing policy")
        mimicpolicy.visualize(env)


if __name__ == '__main__':
    main()
