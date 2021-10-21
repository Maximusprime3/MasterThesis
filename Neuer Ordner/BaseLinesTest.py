# ------------------------------------------------------------------------------------------------
import gym
from stable_baselines3 import A2C

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




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='malmovnv test')
    parser.add_argument('--mission', type=str, default='missions/mobchase_single_agent.xml', help='the mission xml')
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

    def run(role):
        env = malmoenv.make()
        env.init(xml,
                 args.port, server=args.server,
                 server2=args.server2, port2=(args.port + role),
                 role=role,
                 exp_uid=args.experimentUniqueId,
                 episode=args.episode, resync=args.resync)

        def makeEnv():
            menv = malmoenv.make()
            menv.init(xml,
                     args.port, server=args.server,
                     server2=args.server2, port2=(args.port + role),
                     role=role,
                     exp_uid=args.experimentUniqueId,
                     episode=args.episode, resync=args.resync)
            return menv

        def log(message):
            print('[' + str(role) + '] ' + message)


        vecenv = DummyVecEnv([lambda: makeEnv()])

        print('make model')
        model = A2C('MlpPolicy', vecenv, verbose=1)
        print('start train')
        model.learn(total_timesteps=10)
        print('trained')

        safe = True
        once = True
        video = []
        for r in range(args.episodes):
            log("reset " + str(r))
            vecenv.reset()
            action = vecenv.action_space.sample()
            log("action: " + str(action))
            obs, reward, done, info = vecenv.step([action])
            obs = np.vectorize(np.array(obs))
            print('OBS', np.array(obs).shape)
            done = False
            steps = 0
            while not done:
                action, _state = model.predict(obs, deterministic=True)
                log("action: " + str(action))
                obs, reward, done, info = vecenv.step([action])
                if safe:
                    steps += 1
                    safe = False
                    #
                    img = Image.fromarray(obs)
                    video.append(img)
                    img.save('TESTimage.png')
                    print('IMAGESHAPE', img.shape())
                    #obss = pd.DataFrame([obs, reward, done, info])
                    #obss.to_csv("observations")

                log("reward: " + str(reward))
                # log("done: " + str(done))
                # log("info: " + str(info))
                #log(" obs: " + str(obs))

                time.sleep(.05)
            #obs = vecenv.reset()
            if once:
                video = np.array(video)
                print('VIDEOSHAPE', video.shape())
                once = False
        vecenv.close()

    threads = [Thread(target=run, args=(i,)) for i in range(number_of_agents)]

    [t.start() for t in threads]
    [t.join() for t in threads]










