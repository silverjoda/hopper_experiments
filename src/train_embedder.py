from policy import Policy
from utils import Logger, Scaler
import time

from train import init_gym, run_episode
from networks import *


def train_embedder(mimicpolicy,
                masterpolicy,
                env,
                scaler,
                n_episodes=100):

    # Get parameters from policy
    seq_len = mimicpolicy.seq_len
    n_units_list = mimicpolicy.n_units_list

    for i in range(n_episodes):
        observes, actions, _, _ = run_episode(env,
                                            masterpolicy,
                                            scaler,
                                            animate=False)

        state_list = []
        for n in n_units_list:
            st_c = np.zeros((1, n))
            st_h = np.zeros((1, n))
            state_list.append((st_c, st_h))

        state_list = mimicpolicy.run(observes[:seq_len], state_list)

        ep_mse = 0
        for j in range(1, int(len(observes)/seq_len)):
            mse, state_list = mimicpolicy.train(
            observes[seq_len * j:seq_len * j + seq_len], state_list)
            ep_mse += mse

        print("Episode {}/{}, mse: {}".format(i, n_episodes,
                                            ep_mse/int(len(observes)/seq_len)))

def main():

    RESTORE_EMBEDDER = False
    TRAIN = True

    # Open AI gym environment
    env, obs_dim, act_dim = init_gym('Hopper-v1')

    # Make master policy which was trained by R
    masterpolicy = Policy(obs_dim, act_dim, 0.003, stochastic_policy=False)
    masterpolicy.restore_weights()
    scaler = Scaler(obs_dim)

    print("Loaded masterpolicy. ")

    # Make mimic policy
    seq_len = 12
    z_dim = 3
    n_units_list = [12,6,6,12] # Last element must be act_dim
    dropout_list = [(1,1),(1,1),(1,1),(1,1)]
    learning_rate = 3e-3
    n_episodes = 300
    embedder = Hembedder(obs_dim,
                         z_dim,
                         n_units_list,
                         dropout_list,
                         seq_len,
                         learning_rate)

    print("Created mimicpolicy")

    if RESTORE_EMBEDDER:
        embedder.restore_weights()
        print("Restored weights")

    if TRAIN:
        # Train the mimic by master
        t1 = time.time()
        train_embedder(embedder,
                       masterpolicy,
                       env,
                       scaler,
                       n_episodes=n_episodes)

        print("Training time taken: {}".format(time.time() - t1))
        embedder.save_weights()
        print("Saved mimic weights")



if __name__ == '__main__':
    main()
