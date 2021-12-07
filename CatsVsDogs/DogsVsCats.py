import os, shutil
import tensorflow
from keras import layers
from keras import models
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt

#preparing directorys
base_dir = os.getcwd()
base_dir = os.path.join(base_dir, 'subCatalog')

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

# uncoment if you havent get proper catalogs with data
# os.makedirs(train_dir)
# os.makedirs(validation_dir)
# os.makedirs(test_dir)
# os.makedirs(train_cats_dir)
# os.makedirs(train_dogs_dir)
# os.makedirs(validation_cats_dir)
# os.makedirs(validation_dogs_dir)
# os.makedirs(test_cats_dir)
# os.makedirs(test_dogs_dir)
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(10000)]
# for fname in fnames:
#     src = os.path.join(original_dir, fname)
#     dst = os.path.join(train_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(10000)]
# for fname in fnames:
#     src = os.path.join(original_dir, fname)
#     dst = os.path.join(train_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(10000, 11500)]
# for fname in fnames:
#     src = os.path.join(original_dir, fname)
#     dst = os.path.join(validation_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(10000, 11500)]
# for fname in fnames:
#     src = os.path.join(original_dir, fname)
#     dst = os.path.join(validation_dogs_dir, fname)
#     shutil.copyfile(src, dst)
#
#
# fnames = ['cat.{}.jpg'.format(i) for i in range(11500, 12000)]
# for fname in fnames:
#     src = os.path.join(original_dir, fname)
#     dst = os.path.join(test_cats_dir, fname)
#     shutil.copyfile(src, dst)
#
#
# fnames = ['dog.{}.jpg'.format(i) for i in range(11500, 12000)]
# for fname in fnames:
#     src = os.path.join(original_dir, fname)
#     dst = os.path.join(test_dogs_dir, fname)
#     shutil.copyfile(src, dst)

#preparing model

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.6))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.RMSprop(lr=1e-4), metrics=['acc'])

#preparing data

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=35,
    class_mode='binary')

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=35,
    class_mode='binary')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=150,
    validation_data=validation_generator,
    validation_steps=40,)

model.save('DogsVsCats_big_2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Accuracy')
plt.plot(epochs, val_acc, 'b', label='Val_Accuracy')
plt.title('Accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Loss')
plt.plot(epochs, val_loss, 'b', label='Val_Loss')
plt.title('Loss')
plt.legend()

plt.show()
