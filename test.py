import pandas as pd
import gym
import keras
import numpy as np
import collections
from keras.models import Sequential
from keras.layers import Dense
import os
import random
from keras.optimizers import Adam
from termcolor import colored
import pickle


os.environ["CUDA_VISIBLE_DEVICES"]="-1"
env = gym.make("LunarLander-v2")

gamma = 0.99
seed = 1255
episodes = 10000
np.random.seed(seed)
env.seed(seed)
action_space = env.action_space.n
observation_space = env.observation_space.shape[0]

def get_action(model, state,  epsilon):
    state = np.array([state])
    result = model.predict(state)
    action_index = np.argmax(result)
    return result, action_index

def createQNetwork(action_space, observation_space):
    model = Sequential()
    model.add(Dense(512, input_dim=observation_space, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_space))
    model.compile(loss='mse', optimizer=Adam(lr= 0.0001), metrics=['accuracy'])
    return model



def load_models():
    try:
        model = pickle.load(open('model32.pkl', 'rb'))
        target_model = pickle.load(open('target_model32.pkl', 'rb'))
    except FileNotFoundError as e:
        model = createQNetwork(action_space, observation_space)
        target_model = createQNetwork(action_space, observation_space)

    return model, target_model

def save_stats(array):
    stats_array = np.loadtxt("stats32.txt")
    array = np.append(stats_array, array)

    stats_file = open("stats32.txt", "w")
    np.savetxt(stats_file, array)
    stats_file.close()


def test():
    stats = np.array([])
    model, target_model = load_models()
    
    for eee in range(episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:

            pre, action = get_action(model, state, epsilon)
            # next step
            new_state, reward, done, _ = env.step(action)
            score += reward 

            if done:
                stats = np.append(stats, [score]) 
                print("*****************************")
                print("episode: " + str(eee))
                if score > 0:
                    print(colored(score, 'yellow'))
                else:
                    print(colored(score, 'red'))

        if (eee%100 == 0):
        save_stats(stats)  
        stats = np.array([])

test()