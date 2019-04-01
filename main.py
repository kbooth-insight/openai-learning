# import necessary modules from keras
from keras.layers import Dense
from keras.models import Sequential

from datetime import datetime
from keras import callbacks

from easy_tf_log import tflog
from karapathy import prepro, discount_rewards
import os
import time
import tensorflow


# creates a generic neural network architecture
model = Sequential()

# hidden layer takes a pre-processed frame as input, and has 200 units
model.add(Dense(units=200, input_dim=80*80, activation='relu', kernel_initializer='glorot_uniform'))

# output layer
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal'))

# compile the model using traditional Machine Learning losses and optimizers
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

import numpy as np
import gym


# gym initialization
env = gym.make("Pong-v0")
observation = env.reset()
prev_input = None

# Macros
UP_ACTION = 2
DOWN_ACTION = 3

# Hyperparameters
gamma = 0.99

# initialization of variables used in the main loop
x_train, y_train, rewards = [], [], []
reward_sum = 0
episode_nb = 0


# initialize variables
resume = True
running_reward = None
epochs_before_saving = 10
log_dir = './log' + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

# load pre-trained model if exist
if (resume and os.path.isfile('my_model_weights.h5')):
    print("loading previous weights")
    model.load_weights('my_model_weights.h5')

# add a callback tensorboard object to visualize learning
tbCallBack = callbacks.TensorBoard(histogram_freq=0,
                                   write_graph=True, write_images=True)


highest_score = -21

# main loop
while (True):

    # preprocess the observation, set input as difference between images
    cur_input = prepro(observation)
    x = cur_input - prev_input if prev_input is not None else np.zeros(80 * 80)
    prev_input = cur_input

    # forward the policy network and sample action according to the proba distribution
    proba = model.predict(np.expand_dims(x, axis=1).T)
    action = UP_ACTION if np.random.uniform() < proba else DOWN_ACTION
    y = 1 if action == 2 else 0  # 0 and 1 are our labels

    # log the input and label to train later
    x_train.append(x)
    y_train.append(y)

    # do one step in our environment
    observation, reward, done, info = env.step(action)
    #env.render()
    #time.sleep(0.01)
    rewards.append(reward)
    reward_sum += reward

    # end of an episode
    if done:

        print('At the end of episode', episode_nb, 'the total reward was :', reward_sum)

        # increment episode number
        episode_nb += 1

        highest_score = reward_sum
        # training
        model.fit(x=np.vstack(x_train), y=np.vstack(y_train), verbose=1, callbacks=[tbCallBack],
                  sample_weight=discount_rewards(rewards, gamma))

        # Saving the weights used by our model
        if episode_nb % epochs_before_saving == 0:
            model.save_weights('my_model_weights.h5')

        # Log the reward
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        tflog('running_reward', running_reward)
        tflog('score', reward_sum)
        tflog('total_steps', len(x_train))

        # Reinitialization
        x_train, y_train, rewards = [], [], []
        observation = env.reset()
        reward_sum = 0
        prev_input = None