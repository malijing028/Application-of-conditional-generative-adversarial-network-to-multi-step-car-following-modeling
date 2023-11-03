import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pickle import load
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import LSTM, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, ELU, ReLU, RepeatVector, TimeDistributed, Reshape
from tensorflow.keras import Sequential, regularizers
from tensorflow.python.client import device_lib
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model


# Load data
X_train = np.load("X_train.npy", allow_pickle=True)
y_train = np.load("y_train.npy", allow_pickle=True)
yc_train = np.load("yc_train.npy", allow_pickle=True)


# Define the generator
def Generator(input_dim, output_dim, feature_size_in, feature_size_out, img_shape) -> tf.keras.models.Model:
    model = Sequential()
    model.add(LSTM(32, activation='tanh', input_shape=(input_dim, feature_size_in)))
    model.add(RepeatVector(output_dim))
    model.add(LSTM(32, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(feature_size_out)))
    model.add(Reshape(img_shape))
    return model


# Define the discriminator
def Discriminator(xy_shape) -> tf.keras.models.Model:
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=xy_shape))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(64))
    #model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model


class GAN():
    def __init__(self, generator, discriminator, opt):
        self.opt = opt
        self.lr = opt["lr"]
        self.generator = generator
        self.discriminator = discriminator
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.batch_size = self.opt['bs']
        self.checkpoint_dir = '../training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer,
                                              generator=self.generator,
                                              discriminator=self.discriminator)

        model_json = generator.to_json()
        with open("generator_model.json", "w") as json_file:
            json_file.write(model_json)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    #@tf.function
    def train_step(self, data, idx_0):
        X_train, y_train, yc = data
        # Get a random batch of real images
        idx = np.arange(idx_0, idx_0+self.batch_size, 1)
        real_input = X_train[idx]
        real_y = y_train[idx]
        yc = yc_train[idx]

        # Train the discriminator
        with tf.GradientTape() as disc_tape:
            # generate fake output
            generated_data = self.generator(real_input, training=True)
            # reshape the data
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            d_fake_input = tf.concat([tf.cast(yc, tf.float32), generated_data_reshape], axis=1)
            real_y_reshape = tf.reshape(real_y, [real_y.shape[0], real_y.shape[1], 1])
            d_real_input = tf.concat([tf.cast(yc, tf.float32), tf.cast(real_y_reshape, tf.float32)], axis=1)

            # Reshape for MLP
            # d_fake_input = tf.reshape(d_fake_input, [d_fake_input.shape[0], d_fake_input.shape[1]])
            # d_real_input = tf.reshape(d_real_input, [d_real_input.shape[0], d_real_input.shape[1]])

            fake_output = self.discriminator(d_fake_input, training=True)
            real_output = self.discriminator(d_real_input, training=True)

            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            # generate fake output
            generated_data = self.generator(real_input, training=True)
            # reshape the data
            generated_data_reshape = tf.reshape(generated_data, [generated_data.shape[0], generated_data.shape[1], 1])
            g_fake_input = tf.concat([tf.cast(yc, tf.float32), generated_data_reshape], axis=1)
            g_fake_output = self.discriminator(g_fake_input, training=True)
            gen_loss = self.generator_loss(g_fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return disc_loss, gen_loss

    def train(self, X_train, y_train, yc, opt):
        data = X_train, y_train, yc
        idx_range = np.arange(0, X_train.shape[0]-self.batch_size, 1)
        idx_0 = 0
        epochs = opt["epoch"]
        losses = []

        for epoch in range(epochs):
            #start = time.time()
            if idx_0 >= idx_range[-1]+1:
                idx_0 = 0
            d_loss, g_loss = self.train_step(data, idx_0)
            idx_0 += 128

            # For printing loss
            if (epoch + 1) % int(X_train.shape[0]/self.batch_size) == 0:
            losses.append((epoch+1, d_loss.numpy(), g_loss.numpy()))

            # Save the model every 10 epochs
            if (epoch + 1) % 100 == 0:
                tf.keras.models.save_model(generator, 'generator_%d.h5' % (epoch+1))
                print('Epoch %d/%d', %(epoch+1, epochs), 'd_loss', d_loss.numpy(), 'g_loss', g_loss.numpy())
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        losses_pd = pd.DataFrame(np.array(losses))
        losses_pd.to_csv('losses.csv',index=False,sep=',')
        return


if __name__ == '__main__':
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    feature_size_in = X_train.shape[2]
    feature_size_out = yc_train.shape[2]
    img_shape = (output_dim, feature_size_out)
    xy_shape = (input_dim+output_dim, feature_size_out)

    ## For Bayesian
    opt = {"lr": 0.00005, "epoch": 50000, 'bs': 128}

    generator = Generator(input_dim, output_dim, feature_size_in, feature_size_out, img_shape)
    discriminator = Discriminator(xy_shape)
    gan = GAN(generator, discriminator, opt)
    gan.train(X_train, y_train, yc_train, opt)
