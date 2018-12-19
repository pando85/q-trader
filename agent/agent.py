from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam, SGD

import numpy as np
import random
from collections import deque


class Agent:
  def __init__(self, state_size, is_eval=False, model_name=""):
    self.state_size = state_size  # normalized previous days
    self.action_size = 3  # sit, buy, sell
    self.memory = deque(maxlen=1000)
    self.inventory = []
    self.model_name = model_name
    self.is_eval = is_eval

    self.gamma = 0.995
    self.epsilon = 1.0
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995

    self.model = load_model("models/" + model_name) if is_eval else self._model()

  def _model(self):
    model = Sequential()
    model.add(Dense(units=64, input_dim=(self.state_size + 1), activation="relu"))
    model.add(Dense(units=32, activation="relu"))
    model.add(Dense(units=8, activation="relu"))
    model.add(Dense(self.action_size, activation="linear"))
    # model.compile(loss="mse", optimizer=Adam(lr=0.001))
    model.compile(loss="mse", optimizer=SGD(lr=0.001))

    return model

  def act(self, state):
    if not self.is_eval and np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)

    options = self.model.predict(state)
    return np.argmax(options[0])

  def modify_state(self, state):
    if len(self.inventory) > 0:
      state = np.hstack((state, [[1]]))
    else:
      state = np.hstack((state, [[0]]))

    return state

  def expReplay(self, batch_size):
    mini_batch = []

    # Sequential sampling
    # l = len(self.memory)
    # for i in range(l - batch_size + 1, l):
    #   mini_batch.append(self.memory.popleft())

    # Random sampling
    # subsamples = random.sample(
    #   list(self.memory), min(batch_size * 5, len(self.memory))
    # )
    subsamples = random.sample(list(self.memory), len(self.memory))

    states, targets = [], []
    for state, action, reward, next_state, done in subsamples:
      target = reward
      if not done:
        target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

      target_f = self.model.predict(state)
      target_f[0][action] = target

      states.append(state)
      targets.append(target_f)

    self.model.fit(np.vstack(states), np.vstack(targets), epochs=1, verbose=0,
                   batch_size=batch_size)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
