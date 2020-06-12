# %%
# Load dependencies
import numpy as np
import matplotlib.pyplot as plt
import audio_dspy as adsp
import scipy.signal as signal
import tensorflow as tf
from tensorflow import keras
import librosa
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

NUM_FILES = 400
NUM_SAMPLES = 20000
clean_data = []
fs_data = []
for i in tqdm(range(NUM_FILES)):
    fs = np.random.uniform(48000, 128000)
    x = load_fma_file(files, filepath, fs, NUM_SAMPLES)

    fs_data.append(np.ones_like(x) * (1.0 / fs))
    clean_data.append(x)

clean_data = np.asarray(clean_data)
print(np.shape(clean_data))

# %%
# look at file
idx = 4
plt.plot(clean_data[idx])

# %%
NUM_FILES = len(clean_data)
hyst_data = []
for i, x in tqdm(enumerate(clean_data)):
    fs = 1.0 / fs_data[i][0]
    hyst = adsp.Hysteresis(1.0, 1.0, 1.0, fs, mode='RK4')
    y = hyst.process_block(x)
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
NUM_TRAIN = 350
NUM_VAL = 23
x_data = np.stack((clean_data, fs_data), axis=1)
print(x_data.shape)

x_train, x_val = np.split(x_data, [NUM_TRAIN])
y_train, y_val  = np.split(hyst_data,  [NUM_TRAIN])

# %%
OUT_train  = np.reshape(y_train, (NUM_TRAIN, NUM_SAMPLES, 1))
OUT_val    = np.reshape(y_val, (NUM_VAL, NUM_SAMPLES, 1))
IN_train = np.reshape(x_train.transpose((0, 2, 1)), (NUM_TRAIN, NUM_SAMPLES, 2))
IN_val   = np.reshape(x_val.transpose((0, 2, 1)), (NUM_VAL, NUM_SAMPLES, 2))

print(np.shape(IN_train))

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
model.train(20, IN_train, OUT_train, IN_val, OUT_val)

# %%
# plot metrics
plt.figure()
model.plot_loss()

plt.figure()
model.plot_error()

# %%
# Test prediction
idx = 100
print(np.shape(x_data[idx]))

predictions = model.model.predict(IN_train[idx].reshape(1, NUM_SAMPLES, 2)).flatten()
print(np.shape(predictions))

# Plot the predictions along with the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(hyst_data[idx], 'c', label='Actual')
plt.plot(predictions, 'r--', label='Predicted')
plt.legend()
plt.xlim(0, 3000)
plt.xlabel('Time [samples]')

# %%
freqs, pred_fft = plot_fft(predictions, 1.0 / fs_data[idx][0])
freqs, target_fft = plot_fft(hyst_data[idx], 1.0 / fs_data[idx][0])

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
idx = 100
start = 5500
end = 6000
out_data = model.model.predict(IN_train[idx].reshape(1, NUM_SAMPLES, 2)).flatten()
plt.plot(clean_data[idx][start:end], hyst_data[idx][start:end])
plt.plot(clean_data[idx][start:end], out_data[start:end], '--')

# %%
model.save_model('models/hysteresis_fs.json')

# %%
