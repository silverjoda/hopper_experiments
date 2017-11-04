from policy import Policy
from utils import Logger, Scaler
import time

from train import init_gym, run_episode
from networks import *

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def train_qnet(qnet,
                masterpolicy,
                env,
                scaler,
                n_episodes=100,
                batchsize=32):

    for i in range(n_episodes):
        observes, actions, rewards, _ = run_episode(env,
                                            masterpolicy,
                                            scaler,
                                            animate=False)

        mse = qnet.train(observes, actions, rewards, batchsize)
        print("Episode {}/{}, mse: {}".format(i, n_episodes, mse))


def main():

    RESTORE_QNET = False
    TRAIN = True
    VISUALIZE_QNET = True
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

    qnet = SimpleQNet(obs_dim, act_dim, 0.99)

    if RESTORE_QNET:
        qnet.restore_weights()

    if TRAIN:
        # Train the qnet by master
        t1 = time.time()
        train_qnet(qnet,
                    masterpolicy,
                    env,
                    scaler,
                    n_episodes=300,
                    batchsize=64)
        print("Training time taken: {}".format(time.time() - t1))
        qnet.save_weights()
        print("Saved qnet weights")


    if VISUALIZE_QNET:
        # Visualise the policy
        print("Visualizing qnet policy")
        qnet.visualize(env)


if __name__ == '__main__':
    main()
