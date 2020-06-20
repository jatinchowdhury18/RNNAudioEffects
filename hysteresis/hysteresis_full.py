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

NUM_FILES = 1600
NUM_SAMPLES = 16000
QUIET_THRESH = NUM_SAMPLES / 100
clean_data = []
fs_data = []
for i in tqdm(range(NUM_FILES)):
    fs = np.random.uniform(88000, 192000)
    x = load_fma_file(files, filepath, fs, NUM_SAMPLES)

    # skip if too quiet
    if np.sum(np.abs(x)) < QUIET_THRESH:
        i -= 1
        continue

    fs_data.append(np.ones_like(x) * (1.0 / fs))
    clean_data.append(adsp.normalize(x))

clean_data = np.asarray(clean_data)

# %%
# look at file
idx = 8
plt.plot(clean_data[idx])

# %%
hyst_data = []
drive_data = []
sat_data = []
width_data = []
for i, x in tqdm(enumerate(clean_data)):
    fs = 1.0 / fs_data[i][0]
    drive = np.random.uniform()
    sat = np.random.uniform()
    width = np.random.uniform()
    hyst = adsp.Hysteresis(drive, sat, width, fs, dAlpha=0.95, mode='RK4')
    y = hyst.process_block(x)

    drive_data.append(np.ones_like(x) * drive)
    sat_data.append(np.ones_like(x) * sat)
    width_data.append(np.ones_like(x) * width)
    hyst_data.append(y.astype(np.float32))

# %%
idx = 4
plt.figure()
plt.plot(clean_data[idx])
plt.plot(hyst_data[idx])

plt.figure()
freqs, x_fft = plot_fft(clean_data[idx], 1.0 / fs_data[idx][0])
freqs, y_fft = plot_fft(hyst_data[idx], 1.0 / fs_data[idx][0])
plt.semilogx(freqs, x_fft)
plt.semilogx(freqs, y_fft)

# %%
NUM_TRAIN = 1550
NUM_VAL = 24
x_data = np.stack((clean_data, drive_data, sat_data, width_data, fs_data), axis=1)

x_train, x_val = np.split(x_data, [NUM_TRAIN])
y_train, y_val  = np.split(hyst_data,  [NUM_TRAIN])

# %%
OUT_train  = np.reshape(y_train, (NUM_TRAIN, NUM_SAMPLES, 1))
OUT_val    = np.reshape(y_val, (NUM_VAL, NUM_SAMPLES, 1))
IN_train = np.reshape(x_train.transpose((0, 2, 1)), (NUM_TRAIN, NUM_SAMPLES, 5))
IN_val   = np.reshape(x_val.transpose((0, 2, 1)), (NUM_VAL, NUM_SAMPLES, 5))

# %%
plt.plot(IN_train[0, :, 0])
plt.plot(IN_train[0, :, 1])

print(IN_train.dtype)
print(OUT_train.dtype)

# %%
np.save("data/out_train.npy", OUT_train)
np.save("data/out_val.npy", OUT_val)
np.save("data/in_train.npy", IN_train)
np.save("data/in_val.npy", IN_val)

# %%
OUT_train = np.load("data/out_train.npy")
OUT_val   = np.load("data/out_val.npy")
IN_train  = np.load("data/in_train.npy")
IN_val    = np.load("data/in_val.npy")

NUM_SAMPLES = 16000

# %%
model_file = 'models/hysteresis_full.json'
model_hist = 'models/hysteresis_full_history.txt'

# %%
def model_loss(target_y, predicted_y):
    return losses.esr_loss(target_y, predicted_y, losses.pre_emphasis_filter) + losses.dc_loss(target_y, predicted_y)

# construct model
model = Model(model_loss, optimizer=keras.optimizers.Adam(learning_rate=5.0e-4))
# model.model.add(keras.layers.InputLayer(input_shape=(None, 5)))
# model.model.add(keras.layers.TimeDistributed(keras.layers.Dense(8, activation='tanh')))
# model.model.add(keras.layers.GRU(units=16, return_sequences=True))
# model.model.add(keras.layers.Dense(1))
model.load_model(model_file)
model.load_history(model_hist)

model.model.summary()

# %%
model.train(100, IN_train, OUT_train, IN_val, OUT_val, save_model=model_file, save_hist=model_hist)
# model.train_until(0.01, IN_train, OUT_train, IN_val, OUT_val)

# %%
# plot metrics
plt.figure()
model.plot_loss()
plt.ylim(0, 0.1)

plt.figure()
model.plot_error()

# %%
# Test prediction
idx = 44
predictions = model.model.predict(IN_train[idx].reshape(1, NUM_SAMPLES, 5)).flatten()

# Plot the predictions along with the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(OUT_train[idx], 'c', label='Actual')
plt.plot(predictions, 'r--', label='Predicted')
plt.legend()
plt.xlim(0, 3000)
plt.xlabel('Time [samples]')

# %%
fs = 1.0 / IN_train[idx,0,-1]
freqs, pred_fft = plot_fft(predictions, fs)
freqs, target_fft = plot_fft(OUT_train[idx], fs)

# Plot the predictions along with to the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.semilogx(target_fft, 'b', label='Actual')
plt.semilogx(pred_fft, 'r--', label='Predicted')
plt.legend()
plt.xlim(50, 20000)
plt.ylim(-5)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')

# %%
start = 5500
end = 7000
plt.plot(clean_data[idx][start:end], hyst_data[idx][start:end])
plt.plot(clean_data[idx][start:end], predictions[start:end], '--')

# %%
model.save_model(model_file)
model.save_history(model_hist)

# %%
