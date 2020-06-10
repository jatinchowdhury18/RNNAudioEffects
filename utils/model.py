import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

class Model():
    def __init__(self, loss_func, optimizer=keras.optimizers.Adam()):
        self.model = keras.Sequential()
        self.opt = optimizer
        self.loss_func = loss_func

        self.train_loss = []
        self.train_err = []
        self.val_loss = []
        self.val_err = []

    def train(self, num_epochs, in_train, out_train, in_val=None, out_val=None, N_skip=1000, N_block=2048):
        N_samples = in_train.shape[1]
        for epoch in range(num_epochs):
            epoch_loss_avg = keras.metrics.Mean()
            epoch_error = keras.metrics.MeanSquaredError()
            val_loss_avg = keras.metrics.Mean()
            val_error = keras.metrics.MeanSquaredError()

            # run training
            self.model.reset_states() # clear existing state
            self.model(in_train[:, :N_skip, :]) # process some samples to build up state

            # iterate over blocks
            for n in range(N_skip, N_samples-N_block, N_block):
                # compute loss
                with tf.GradientTape() as tape:
                    y_pred = self.model(in_train[:, n:n+N_block, :])
                    loss = self.loss_func(out_train[:, n:n+N_block, :], y_pred)

                # apply gradients
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

                # update training metrics
                epoch_loss_avg.update_state(loss)
                epoch_error.update_state(out_train[:, n:n+N_block, :], y_pred)

            # run validation
            if in_val is not None:
                self.model.reset_states()
                y_val = self.model(in_val)
                loss = self.loss_func(out_val, y_val)
                val_loss_avg.update_state(loss)
                val_error.update_state(out_val, y_val)
            
            # end epoch
            self.train_loss.append(epoch_loss_avg.result())
            self.train_err.append(epoch_error.result())
            self.val_loss.append(val_loss_avg.result())
            self.val_err.append(val_error.result())
            print("Epoch {:03d}: Loss: {:.3f}, Error: {:.3%}, Val_Loss: {:.3f}, Val_Error: {:.3%}".format(
                epoch+1, epoch_loss_avg.result(), epoch_error.result(), val_loss_avg.result(), val_error.result()))
        
        print("DONE!")


    def plot_loss(self):
        epochs = range(1, len(self.train_loss) + 1)
        plt.plot(epochs, self.train_loss, 'g.', label='Training Loss')
        plt.plot(epochs, self.val_loss, 'b', label='Validation Loss')
        plt.title('Training and validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    def plot_error(self):
        epochs = range(1, len(self.train_loss) + 1)
        plt.plot(epochs, self.train_err, 'g.', label='Training Error')
        plt.plot(epochs, self.val_err, 'b', label='Validation Error')
        plt.title('Training and validation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.legend()
