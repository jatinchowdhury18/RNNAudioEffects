# %%
# Load dependencies
import tensorflow as tf
from tensorflow import keras
import librosa

import numpy as np
import matplotlib.pyplot as plt
import audio_dspy as adsp
import scipy.signal as signal
from tqdm import tqdm
import os
import random
import sys

sys.path.append('..')
from utils.utils import plot_fft, load_fma_file
from utils.model import Model
import utils.losses as losses

# %%
# load files
filepath = '../Data/fma_small/'
files = os.listdir(filepath)

NUM_FILES = 20
NUM_SAMPLES = 20000
FS = 96000
clean_data = []
for i in tqdm(range(NUM_FILES)):
    x = load_fma_file(files, filepath, FS, NUM_SAMPLES)
    clean_data.append(x)

clean_data = np.asarray(clean_data)

# %%
# look at file
idx = 4
plt.plot(clean_data[idx])

# %%
vib_data = []
sine_data = []
for x in tqdm(clean_data):
    freq = np.random.uniform(0.0, 20)
    depth = np.random.uniform()
    sine = depth * np.sin(2 * np.pi * freq * np.arange(len(x)) / FS)
    y = x * sine

    sine_data.append(sine.astype(np.float32))
    vib_data.append(y.astype(np.float32))

# %%
idx = 4
plt.figure()
plt.plot(clean_data[idx])
plt.plot(sine_data[idx])
plt.plot(vib_data[idx])

# %%
NUM_TRAIN = 18
NUM_VAL = 2
x_data = np.stack((clean_data, sine_data), axis=1)

x_train, x_val = np.split(x_data, [NUM_TRAIN])
y_train, y_val  = np.split(vib_data,  [NUM_TRAIN])

# %%
OUT_train  = np.reshape(y_train, (NUM_TRAIN, NUM_SAMPLES, 1))
OUT_val    = np.reshape(y_val, (NUM_VAL, NUM_SAMPLES, 1))
IN_train = np.reshape(x_train.transpose((0, 2, 1)), (NUM_TRAIN, NUM_SAMPLES, 2))
IN_val   = np.reshape(x_val.transpose((0, 2, 1)), (NUM_VAL, NUM_SAMPLES, 2))

# %%
plt.plot(IN_train[0, :, 0])
plt.plot(IN_train[0, :, 1])

print(IN_train.dtype)
print(OUT_train.dtype)

# %%
def model_loss(target_y, predicted_y):
    return losses.esr_loss(target_y, predicted_y, losses.pre_emphasis_filter) + losses.dc_loss(target_y, predicted_y)

# construct model
model = Model(model_loss, optimizer=keras.optimizers.Adam(learning_rate=5.0e-4))
model.model.add(keras.layers.InputLayer(input_shape=(None, 2)))
model.model.add(keras.layers.TimeDistributed(keras.layers.Dense(8, activation='tanh')))
model.model.add(keras.layers.GRU(units=16, return_sequences=True))
model.model.add(keras.layers.Dense(1))

model.model.summary()

# %%
model.train(100, IN_train, OUT_train, IN_val, OUT_val)

# %%
# plot metrics
plt.figure()
model.plot_loss()

plt.figure()
model.plot_error()

# %%
# Test prediction
idx = 15

predictions = model.model.predict(IN_train[idx].reshape(1, NUM_SAMPLES, 2)).flatten()

# Plot the predictions along with the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(vib_data[idx], 'c', label='Actual')
plt.plot(predictions, 'r--', label='Predicted')
plt.legend()
plt.xlim(0, 3000)
plt.xlabel('Time [samples]')

# %%
model.save_model('models/vibrato.json')
model.save_history('models/vibrato_history.txt')

# %%
