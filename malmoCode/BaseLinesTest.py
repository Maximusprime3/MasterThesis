# ------------------------------------------------------------------------------------------------
import gym
from stable_baselines3 import SAC
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_checker import check_env
import stable_baselines3
import malmoenv
import argparse
from pathlib import Path
import time
from lxml import etree
from threading import Thread
from PIL import Image
import numpy as np
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
#for the custom policy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import gym
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn import CnnPolicy

#To start minecraft use
    # python -c "import malmoenv.bootstrap; malmoenv.bootstrap.launch_minecraft(9000)"
#To start a script with minecraft use another terminal navigate to malmoplatform/malmoenv and use
    # python BaseLinesTest.py

#This a custom policy for stable baselines3
class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super(CustomNetwork, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.policy_net(features), self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):

        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        self.ortho_init = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malmovnv test')
    parser.add_argument('--mission', type=str, default='missions/sticktest.xml', help='the mission xml')
    parser.add_argument('--port', type=int, default=9000, help='the mission server port')
    parser.add_argument('--server', type=str, default='127.0.0.1', help='the mission server DNS or IP address')
    parser.add_argument('--server2', type=str, default=None, help="(Multi-agent) role N's server DNS or IP")
    parser.add_argument('--port2', type=int, default=9000, help="(Multi-agent) role N's mission port")
    parser.add_argument('--episodes', type=int, default=1, help='the number of resets to perform - default is 1')
    parser.add_argument('--episode', type=int, default=0, help='the start episode - default is 0')
    parser.add_argument('--resync', type=int, default=0, help='exit and re-sync on every N - default 0 meaning never')
    parser.add_argument('--experimentUniqueId', type=str, default='test1', help="the experiment's unique id.")

    args = parser.parse_args()
    if args.server2 is None:
        args.server2 = args.server

    xml = Path(args.mission).read_text()

    mission = etree.fromstring(xml)
    number_of_agents = len(mission.findall('{http://ProjectMalmo.microsoft.com}AgentSection'))
    print("number of agents: " + str(number_of_agents))
    print('mission loaded')
    action_filter = []
    grid_obs = True

    def run(role):
        print('role', role)
        env = malmoenv.make()
        env.init(xml,
                 args.port, server=args.server,
                 server2=args.server2, port2=(args.port + role),
                 role=role,
                 exp_uid=args.experimentUniqueId,
                 episode=args.episode, resync=args.resync,
                 action_filter=action_filter, grid_obs=grid_obs)

        def makeEnv():
            menv = malmoenv.make()
            menv.init(xml,
                     args.port, server=args.server,
                     server2=args.server2, port2=(args.port + role),
                     role=role,
                     exp_uid=args.experimentUniqueId,
                     episode=args.episode, resync=args.resync,
                     action_filter=action_filter, grid_obs=grid_obs)
            return menv

        def log(message):
            print('[' + str(role) + '] ' + message)


        vecenv = DummyVecEnv([lambda: makeEnv()])
        #print('check env')
        #check_env(vecenv)
        print('make model')
        j=0
        actionss =[]
        while j < 10000:
            actionss.append(env.action_space.sample())
            j+=1

        print('actions', env.action_space, np.unique(np.array(actionss)))
        obs1, r, do, inf = env.step(env.action_space.sample())
        print('Observation1', inf)
        #d=action_space.sample(1000)
        #model = DQN(CnnPolicy, vecenv, verbose=1) #SAC is best -carl

        #model = PPO('CnnPolicy', vecenv, verbose=1)
        model = PPO("MultiInputPolicy", env, verbose=1)
        print('start train')
        model.learn(total_timesteps=100)
        print('trained')
        model.save("PPO_CNNpolicy_MyStick")

        safe = True
        once = True
        video = []
        for r in range(args.episodes):
            log("reset " + str(r))
            vecenv.reset()
            obs = vecenv.reset()
            #print('obs3', obs.shape, obs)
            done = False
            steps = 0
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                log("action: " + str(action))
                obs, reward, done, info = vecenv.step(action)
                if safe:
                    steps += 1
                    safe = False
                    print('Observation', obs.shape, obs)

                    #
                    '''
                    imgobs = np.array(obs[0]).transpose(1, 0, 2)
                    print('obs running', imgobs.shape, obs)
                    img = Image.fromarray(imgobs)
                    video.append(img)
                    img.save('TESTimage.png')
                    print('IMAGESHAPE', img.shape())
                    #obss = pd.DataFrame([obs, reward, done, info])
                    #obss.to_csv("observations")
                    '''
                log("reward: " + str(reward))
                # log("done: " + str(done))
                # log("info: " + str(info))
                #log(" obs: " + str(obs))

                time.sleep(.05)

            '''
            if once:
                video = np.array(video)
                print('VIDEOSHAPE', video.shape())
                once = False
            '''
        vecenv.close()

    threads = [Thread(target=run, args=(i,)) for i in range(number_of_agents)]

    [t.start() for t in threads]
    [t.join() for t in threads]













