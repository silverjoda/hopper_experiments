from fullsys_rnn.full_system_rnn import RNNSim
import numpy as np
import gym
from ornstein_proc import OrnsteinUhlenbeckActionNoise

def main():
    # Settings
    config = {"policy_lr" : 1e-4,
              "env_lr": 1e-4,
              "n_state" : 48,
              "env_name" : "Hopper-v1",
              "training_iters" : 1000,
              "model_opt_iters" : 8,
              "policy_opt_iters" : 8,
              "n_rollouts" : 8,
              "gpu" : 0,
              "max_steps" : 300,}

    # Make gym environment
    env = gym.make(config['env_name'])
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    config["env"] = env

    # Make our training model
    sys_rnn = RNNSim(config, obs_dim, act_dim)

    ep_actions = []
    ep_observations = []
    ep_rewards = []

    # Train
    for gi in range(config['training_iters']):
        anoise = OrnsteinUhlenbeckActionNoise(np.zeros(act_dim),
                                              sigma=.15,
                                              theta=.15,
                                              dt=1e-2)

        # Gather N rollouts using simulator dynamics and trained policy
        # with random exploration
        longest_ep = 0
        for ri in range(config["n_rollouts"]):

            actions = []
            observations = []
            rewards = []

            obs = env.reset()
            observations.append(obs)

            for si in range(config["max_steps"]):
                action = sys_rnn.predict(obs)
                obs, rew, done, _ = env.step(action + anoise())
                actions.append(action)
                observations.append(obs)
                rewards.append(rew)

                if si > longest_ep:
                    longest_ep = si

                if done:
                    break

            actions = np.array(actions, dtype=np.float32)
            observations = np.array(observations, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)

            ep_actions.append(actions)
            ep_observations.append(observations)
            ep_rewards.append(rewards)

        # Update model dynamics and cost prediction by minimizing MSE
        for mi in range(config['model_opt_iters']):
            rndvec = np.arange(len(ep_actions))
            np.random.shuffle(rndvec)

            total_c_loss = 0
            total_s_loss = 0
            for r in rndvec:
                observations = ep_observations[r]
                actions = ep_actions[r]
                rews = ep_rewards[r]
                c_loss, s_loss = sys_rnn.trainmodel(observations, actions, rews)
                total_c_loss += c_loss
                total_s_loss += s_loss

            print("Glob: {}/{}, model {}/{}, Total c_loss: {}, s_loss: {}".format(
                gi + 1,config['training_iters'],mi + 1,config['model_opt_iters'],
                total_c_loss / len(rndvec),
                total_s_loss / len(rndvec)
            ))

        # Backprop into policy, maximizing predicted reward
        for ai in range(config['policy_opt_iters']):
            predicted_cost = sys_rnn.trainpolicy(longest_ep)

            print("Glob: {}/{}, policy {}/{}, predicted cost: {}".format(
                gi + 1, config['training_iters'], ai + 1, config['model_opt_iters'],
                predicted_cost
            ))


if __name__ == "__main__":
    main()