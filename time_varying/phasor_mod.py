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
from utils.phaser_mod import PhasorMod

# %%
# load files
filepath = '../Data/fma_small/'
files = os.listdir(filepath)

NUM_FILES = 50
NUM_SAMPLES = 15000
# FS = 48000
clean_data = []
fs_data = []
for i in tqdm(range(NUM_FILES)):
    fs = 48000 # np.random.uniform(32000, 96000)
    x = load_fma_file(files, filepath, fs, NUM_SAMPLES)

    clean_data.append(x)
    fs_data.append(np.ones_like(x) * (1.0 / fs))

clean_data = np.asarray(clean_data)
fs_data = np.asarray(fs_data)

# %%
# look at file
idx = 1
plt.plot(clean_data[idx])

# %%
phase_data = []
lfo_data = []
mod_data = []
for i, x in tqdm(enumerate(clean_data)):
    fs = 1.0 / fs_data[i][0]
    mod = np.random.randint(0, 8)

    freq = np.random.uniform(0.0, 20)
    depth = np.random.uniform()
    lfo = depth * np.sin(2 * np.pi * freq * np.arange(len(x)) / fs)

    phasor = PhasorMod(fs)
    y = phasor.process_block(x, lfo, mod)

    lfo_data.append(lfo.astype(np.float32))
    mod_data.append(np.ones_like(x) * mod)
    phase_data.append(y.astype(np.float32))

# %%
idx = 1
plt.figure()
plt.plot(clean_data[idx])
plt.plot(phase_data[idx], '--')
plt.plot(lfo_data[idx])
# plt.plot(mod_data[idx])

print(1.0 / fs_data[idx][0])

# %%
NUM_TRAIN = 48
NUM_VAL = 2
x_data = np.stack((clean_data, lfo_data, mod_data), axis=1)

x_train, x_val = np.split(x_data, [NUM_TRAIN])
y_train, y_val  = np.split(phase_data, [NUM_TRAIN])

# %%
OUT_train  = np.reshape(y_train, (NUM_TRAIN, NUM_SAMPLES, 1))
OUT_val    = np.reshape(y_val, (NUM_VAL, NUM_SAMPLES, 1))
IN_train = np.reshape(x_train.transpose((0, 2, 1)), (NUM_TRAIN, NUM_SAMPLES, 3))
IN_val   = np.reshape(x_val.transpose((0, 2, 1)), (NUM_VAL, NUM_SAMPLES, 3))

# %%
plt.plot(IN_train[0, :, 0])
plt.plot(IN_train[0, :, 1])
plt.plot(IN_train[0, :, 2])

print(IN_train.dtype)
print(OUT_train.dtype)

# %%
model_file = 'models/phasor_mod.json'
model_hist = 'models/phasor_mod_history.txt'

# %%
def model_loss(target_y, predicted_y):
    return losses.esr_loss(target_y, predicted_y, losses.pre_emphasis_filter) + losses.dc_loss(target_y, predicted_y)

# construct model
model = Model(model_loss, optimizer=keras.optimizers.Adam(learning_rate=5.0e-4))
model.model.add(keras.layers.InputLayer(input_shape=(None, 3)))
model.model.add(keras.layers.TimeDistributed(keras.layers.Dense(4, activation='tanh')))
model.model.add(keras.layers.GRU(units=4, return_sequences=True))
model.model.add(keras.layers.GRU(units=4, return_sequences=True))
model.model.add(keras.layers.GRU(units=4, return_sequences=True))
model.model.add(keras.layers.GRU(units=4, return_sequences=True))
model.model.add(keras.layers.Dense(1))
# model.load_model(model_file)
# model.load_history(model_hist)

model.model.summary()

# %%
model.train(50, IN_train, OUT_train, IN_val, OUT_val)

# %%
# plot metrics
plt.figure()
model.plot_loss()

plt.figure()
model.plot_error()

# %%
# Test prediction
idx = 2
predictions = model.model.predict(IN_train[idx].reshape(1, NUM_SAMPLES, 3)).flatten()

# Plot the predictions along with the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(phase_data[idx], 'c', label='Actual')
# plt.plot(clean_data[idx], 'g--', label='clean')
plt.plot(predictions, 'r--', label='Predicted')
plt.legend()
# plt.xlim(0, 3000)
plt.xlabel('Time [samples]')

# %%
fs = 1.0 / fs_data[idx][0]
freqs, pred_fft = plot_fft(predictions, fs)
freqs, target_fft = plot_fft(phase_data[idx], fs)

# Plot the predictions along with to the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.semilogx(freqs, target_fft, 'b', label='Actual')
plt.semilogx(freqs, pred_fft, 'r--', label='Predicted')
plt.legend()
plt.xlim(50, 20000)
plt.ylim(-5)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')

# %%
model.save_model(model_file)
model.save_history(model_hist)

# %%
