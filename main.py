import tensorflow as tf
import gym
from agent import DQNAgent


def learn(env, agent, episodes=100, epochs=100, verbose=False):
    step = 0
    for k in range(episodes):
        R = 0
        for _ in range(epochs):
            s = env.reset()
            done = False
            
            while not done:
                
                a = agent.act(s)
                s_ , r, done, _ = env.step(a)
                agent.store_experience([s, [a, r, done], s_])
                if agent.is_mem_ready():
                    agent.learn()
                s = s_
                
                step += 1
                R += r
        if verbose: print("Episode {} mean reward: {}".format(k, R/epochs))
    
tf.reset_default_graph()
env = gym.make("CartPole-v1")

a = DQNAgent(4, 2)

learn(env, a, verbose=True)