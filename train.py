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
        if np.random.random() < epsilon:  
            return 'random', np.random.randint(4) 
        else:
            state = np.array([state])
            result = model.predict(state)
            action_index = np.argmax(result)
            return result, action_index

def createQNetwork(action_space, observation_space):
    model = Sequential()
    model.add(Dense(32, input_dim=observation_space, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_space))
    model.compile(loss='mse', optimizer=Adam(lr= 0.0001), metrics=['accuracy'])
    return model


def save_stats(array):
    stats_array = np.loadtxt("stats.txt")
    array = np.append(stats_array, array)

    stats_file = open("stats.txt", "w")
    np.savetxt(stats_file, array)
    stats_file.close()



def save_models(model, target_model):
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(target_model, open('target_model.pkl', 'wb'))

def load_models():
    try:
        model = pickle.load(open('model.pkl', 'rb'))
        target_model = pickle.load(open('target_model.pkl', 'rb'))
    except FileNotFoundError as e:
        model = createQNetwork(action_space, observation_space)
        target_model = createQNetwork(action_space, observation_space)

    return model, target_model


def train():
    epsilon = 1

    stats = np.array([])
    D = collections.deque(maxlen=5000)
    model, target_model = load_models()
    
    for eee in range(episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:

            env.render()
            # take action
            if epsilon > 0.1:
                epsilon -= 0.0005 
            pre, action = get_action(model, state, epsilon)
            # next step
            new_state, reward, done, _ = env.step(action)
            score += reward 

            D.append([state, action, reward, new_state, done])
            # sample_D = D
           
            try:
                sample_D = random.sample(D, 100)
            except ValueError as e:
                sample_D = D
            
            targets_q = []
            states_to_train_with = []
            # calculate target
            for i in range(0, len(sample_D)):
                output = target_model.predict(np.array([sample_D[i][3]]))
                is_terminal = sample_D[i][4]
                target_q = model.predict(np.array([sample_D[i][0]]))
                # if terminal, only equals reward
                if is_terminal:
                    target_q[0][sample_D[i][1]] = sample_D[i][2]
                else:
                    target_q[0][sample_D[i][1]] = sample_D[i][2] + gamma * np.max(output)
                    
                targets_q.append(target_q[0])
                states_to_train_with.append(sample_D[i][0])
            
            model.fit(np.array(states_to_train_with), np.array(targets_q), epochs=1, verbose=0)
            
            
            state = new_state

            if done:
                stats = np.append(stats, [score]) 
                print("*****************************")
                print("episode: " + str(eee))
                if score > 0:
                    print(colored(score, 'yellow'))
                else:
                    print(colored(score, 'red'))
     
        target_model.set_weights(model.get_weights()) 

        if (eee%100 == 0):
            save_stats(stats)  
            save_models(model, target_model)  
            stats = np.array([])


train()