import os
import random
import matplotlib.image as mpimg
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras.preprocessing import image
from sklearn import preprocessing

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#preparing directorys
base_dir = os.getcwd()
humans_dir = os.path.join(base_dir, 'humans')
report_dir = os.path.join(os.getcwd(), 'report_humans')

# Define input image dimensions
# Large images take too much time and resources.
img_rows = 32
img_cols = 32
channels = 3
img_shape = (img_rows, img_cols, channels)


##########################################################################
# Given input of noise (latent) vector, the Generator produces an image.
def build_generator():

    # Define your generator network
    # Here we are only using Dense layers. But network can be complicated based
    # on the application. For example, you can use VGG for super res. GAN.
    model = models.Sequential()
    model.add(layers.Conv2D(64, (2, 2), activation='relu', input_shape=img_shape, padding="same"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (2, 2), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (2, 2), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(Dense(np.prod(img_shape), activation='relu'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=img_shape)
    img = model(noise)  # Generated image

    return Model(noise, img)


# Alpha — α is a hyperparameter which controls the underlying value to which the
# function saturates negatives network inputs.
# Momentum — Speed up the training
##########################################################################

# Given an input image, the Discriminator outputs the likelihood of the image being real.
# Binary classification - true or false (we're calling it validity)

def build_discriminator():

    model = models.Sequential()
    model.add(layers.Conv2D(64, (2, 2), activation='relu', input_shape=img_shape, padding="same"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (2, 2), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (2, 2), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


# The validity is the Discriminator’s guess of input being real or not.


# Now that we have constructed our two models it’s time to pit them against each other.
# We do this by defining a training function, loading the data set, re-scaling our training
# images and setting the ground truths.
def train(epochs, batch_size=32, save_interval=50):
    # preparing data
    half_batch = int(batch_size / 2)
    x_train = np.zeros((half_batch, img_rows, img_cols, 3))

    # We then loop through a number of epochs to train our Discriminator by first selecting
    # a random batch of images from our true dataset, generating a set of images from our
    # Generator, feeding both set of images into our Discriminator, and finally setting the
    # loss parameters for both the real and fake images, as well as the combined loss.

    for epoch in range(epochs):

        for i in range(half_batch):

            if((epoch+1)*i) < 7218:
                fname_human = os.path.join(humans_dir, os.listdir(humans_dir)[(epoch+1)*i])
            else:
                index = random.randint(0, 7218)
                fname_human = os.path.join(humans_dir, os.listdir(humans_dir)[index])

            img_human = image.load_img(fname_human, target_size=(img_rows, img_cols))
            x = image.img_to_array(img_human)
            #print(x.shape())
            # x = x.reshape((1,) + x.shape)
            x = x / 255
            x_train[i] = x
            # prediction
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random half batch of real images

        noise = np.random.normal(0, 1, (half_batch, img_rows, img_cols, 3))

        # Generate a half batch of fake images
        gen_imgs = generator.predict(noise)

        # Train the discriminator on real and fake images, separately
        # Research showed that separate training is more effective.
        d_loss_real = discriminator.train_on_batch(x_train, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        # take average loss from real and fake images.
        #
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # And within the same loop we train our Generator, by setting the input noise and
        # ultimately training the Generator to have the Discriminator label its samples as valid
        # by specifying the gradient loss.
        # ---------------------
        #  Train Generator
        # ---------------------
        # Create noise vectors as input for generator.
        # Create as many noise vectors as defined by the batch size.
        # Based on normal distribution. Output will be of size (batch size, 1000)
        noise = np.random.normal(0, 1, (batch_size, img_rows, img_cols, 3))

        # The generator wants the discriminator to label the generated samples
        # as valid (ones)
        # This is where the genrator is trying to trick discriminator into believing
        # the generated image is true (hence value of 1 for y)
        valid_y = np.array([1] * batch_size)  # Creates an array of all ones of size=batch size

        # Generator is part of combined where it got directly linked with the discriminator
        # Train the generator with noise as x and 1 as y.
        # Again, 1 as the output as it is adversarial and if generator did a great
        # job of folling the discriminator then the output would be 1 (true)
        g_loss = combined.train_on_batch(noise, valid_y)

        # Additionally, in order for us to keep track of our training process, we print the
        # progress and save the sample image output depending on the epoch interval specified.
        # Plot the progress

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

        # If at save interval => save generated image samples
        if epoch % save_interval == 0:
            save_imgs(epoch)


# when the specific sample_interval is hit, we call the
# sample_image function. Which looks as follows.

def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, img_rows, img_cols, 3))
    #normalize pictures
    gen_imgs = generator.predict(noise)
    gen_imgs /= gen_imgs.max()
    gen_imgs *= 250
    gen_imgs = gen_imgs.astype(int)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    name = os.path.join(report_dir, f'{epoch}.png')
    print(name)
    fig.savefig(name)
    plt.close()


# This function saves our images for us to view


##############################################################################
# example = os.path.join(dandelions_dir, os.listdir(dandelions_dir)[2])
# img_example = image.load_img(example, target_size=(img_rows, img_cols))
# plt.imshow(img_example)
# plt.show()
# Let us also define our optimizer for easy use later on.
# That way if you change your mind, you can change it easily here
optimizer = tf.keras.optimizers.RMSprop(lr=1e-4)  # Learning rate and momentum.

# Build and compile the discriminator first.
# Generator will be trained as part of the combined model, later.
# pick the loss function and the type of metric to keep track.
# Binary cross entropy as we are doing prediction and it is a better
# loss function compared to MSE or other.
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

# build and compile our Discriminator, pick the loss function

# SInce we are only generating (faking) images, let us not track any metrics.
generator = build_generator()
generator.compile(loss='binary_crossentropy', optimizer=optimizer)

##This builds the Generator and defines the input noise.
# In a GAN the Generator network takes noise z as an input to produce its images.
z = Input(shape=img_shape)  # Our random input to the generator
img = generator(z)

# This ensures that when we combine our networks we only train the Generator.
# While generator training we do not want discriminator weights to be adjusted.
# This Doesn't affect the above descriminator training.
discriminator.trainable = False

# This specifies that our Discriminator will take the images generated by our Generator
# and true dataset and set its output to a parameter called valid, which will indicate
# whether the input is real or not.
valid = discriminator(img)  # Validity check on the generated image

# Here we combined the models and also set our loss function and optimizer.
# Again, we are only training the generator here.
# The ultimate goal here is for the Generator to fool the Discriminator.
# The combined model  (stacked generator and discriminator) takes
# noise as input => generates images => determines validity

combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

train(epochs=200000, batch_size=24, save_interval=10)

# Save model for future use to generate fake images
# Not tested yet... make sure right model is being saved..
# Compare with GAN4

generator.save('HumansGAN_v1.h51')  # Test the model on GAN4_predict...

# Change epochs back to 30K

# Epochs dictate the number of backward and forward propagations, the batch_size
# indicates the number of training samples per backward/forward propagation, and the
# sample_interval specifies after how many epochs we call our sample_image function.
