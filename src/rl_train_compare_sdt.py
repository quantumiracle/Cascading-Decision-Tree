import gym
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
import json
from rl import PPO
from rl import StateNormWrapper

def run(EnvName,
        rl_confs,
        mode=None,
        episodes=1000,
        t_horizon=1000,
        model_path=None,
        log_path=None):
    env = StateNormWrapper(gym.make(EnvName), file_name="./rl/rl.json")  # for state normalization
    env = gym.make(EnvName)    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n  # discrete
    model = PPO(state_dim, action_dim, rl_confs["General"]["policy_approx"], rl_confs[EnvName]["learner_args"], \
        **rl_confs[EnvName]["alg_confs"]).to(torch.device(rl_confs[EnvName]["learner_args"]["device"]))
    print_interval = 20
    if mode == 'test':
        model.load_model(model_path)
    rewards_list = []
    for n_epi in range(episodes):
        s = env.reset()
        done = False
        reward = 0.0
        step = 0
        while not done and step < t_horizon:
            if mode == 'train':
                a, prob = model.choose_action(s)
            else:
                a = model.choose_action(s, Greedy=True)
                # a, prob=model.choose_action(s)

            s_prime, r, done, info = env.step(a)

            if mode == 'test':
                env.render()
            else:
                model.put_data(
                    (s, a, r / 100.0, s_prime, prob[a].item(), done))
                # model.put_data((s, a, r, s_prime, prob[a].item(), done))

            s = s_prime

            reward += r
            step += 1
            if done:
                break
        if mode == 'train':
            model.train_net()
            if n_epi % print_interval == 0 and n_epi != 0:
                # plot(rewards_list)
                np.save(log_path, rewards_list)
                torch.save(model.state_dict(), model_path)
                print("# of episode :{}, reward : {:.1f}, episode length: {}".
                      format(n_epi, reward, step))
        else:
            print(
                "# of episode :{}, reward : {:.1f}, episode length: {}".format(
                    n_epi, reward, step))
        rewards_list.append(reward)
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reinforcement learning training.')
    parser.add_argument('--train',
                        dest='train',
                        action='store_true',
                        default=False)
    parser.add_argument('--test',
                        dest='test',
                        action='store_true',
                        default=False)
                        
    parser.add_argument('--env',
                    dest='EnvName',
                    action='store',
                    default=None)

    parser.add_argument('--method',
                    dest='METHOD',
                    action='store',
                    default=None)

    parser.add_argument('--id',
                    dest='id',
                    action='store',
                    default=0)

    parser.add_argument('--depth',
                    dest='depth',
                    action='store',
                    default=2)

    args = parser.parse_args()
    
    METHOD = args.METHOD     #  one of: 'mlp', 'cdt', 'sdt'

    filename = "./sdt/sdt_rl_train_compare.json"

    with open(filename, "r") as read_file:
        rl_confs = json.load(read_file)  # hyperparameters for rl training

    EnvName = args.EnvName 

    rl_confs[EnvName]["learner_args"]["depth"]=int(args.depth)

    # add id 
    rl_confs[EnvName]["train_confs"]["model_path"] = rl_confs[EnvName]["train_confs"]["model_path"]+'_'+args.depth+'_'+args.id
    rl_confs[EnvName]["train_confs"]["log_path"] = rl_confs[EnvName]["train_confs"]["log_path"]+'_'+args.depth+'_'+args.id

    if args.train:
        run(EnvName,
            rl_confs,
            mode='train',
            **rl_confs[EnvName]["train_confs"])
    if args.test:
        run(EnvName,
            rl_confs,
            mode='test',
            **rl_confs[EnvName]["train_confs"])
