from policy import Policy
from utils import Logger, Scaler
import time

from train import init_gym, run_episode
from networks import *

def evaluate_mimic(mimicpolicy, masterpolicy, scaler, env, n):
    total_mse = 0
    for i in range(n):
        observes, actions, _, _ = run_episode(env,
                                              masterpolicy,
                                              scaler,
                                              animate=False)
        total_mse += mimicpolicy.evaluate(np.expand_dims(observes,0),
                                          np.expand_dims(actions, 0))

    return total_mse/n

def train_mimic(mimicpolicy,
                masterpolicy,
                env,
                scaler,
                n_episodes=100,
                batchsize=32):

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

        ep_mse = 0
        for j in range(int(len(observes)/seq_len)):
            mse, state_list = mimicpolicy.train(
            observes[seq_len * j:seq_len * j + seq_len],
            actions[seq_len * j:seq_len * j + seq_len],
            state_list)

            ep_mse += mse


        print("Episode {}/{}, mse: {}".format(i, n_episodes,
                                            ep_mse/int(len(observes)/seq_len)))

def main():

    RESTORE_MIMIC = False
    TRAIN = True
    VISUALIZE_MIMIC = True

    # Open AI gym environment
    env, obs_dim, act_dim = init_gym('Hopper-v1')

    # Make master policy which was trained by R
    masterpolicy = Policy(obs_dim, act_dim, 0.003, stochastic_policy=False)
    masterpolicy.restore_weights()
    scaler = Scaler(obs_dim)

    print("Loaded masterpolicy. ")

    # Make mimic policy
    seq_len = 8
    n_units_list = [6,6,6]
    learning_rate = 5e-3
    mimicpolicy = RecurrentPolicy(obs_dim,
                                  act_dim,
                                  n_units_list,
                                  seq_len,
                                  learning_rate)
    print("Created mimicpolicy")

    if RESTORE_MIMIC:
        mimicpolicy.restore_weights()
        print("Restored weights")

    if TRAIN:
        # Train the mimic by master
        t1 = time.time()
        train_mimic(mimicpolicy,
                    masterpolicy,
                    env,
                    scaler,
                    n_episodes=1,
                    batchsize=32)
        print("Training time taken: {}".format(time.time() - t1))
        mimicpolicy.save_weights()
        print("Saved mimic weights")

    if VISUALIZE_MIMIC:
        # Visualise the policy
        print("Visualizing policy")
        mimicpolicy.visualize(env)



if __name__ == '__main__':
    main()
