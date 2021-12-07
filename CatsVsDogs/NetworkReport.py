from keras.preprocessing import image
from keras import models
from keras import backend as K
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# old version of tf, could make problems in future
tf.compat.v1.disable_eager_execution()


class NetworkReport:
    def __init__(self, model, reports_directory, models_directory, data_directory, save_frequency, target_size):
        self.model = model
        self.reports_directory = reports_directory
        self.model_directory = models_directory
        self.data_directory = data_directory
        self.save_frequency = save_frequency #ToDo add learning Model
        self.target_size = target_size

        number_of_layers = 0
        for layer in self.model.layers:
            number_of_layers += 1

        self.number_of_layers = number_of_layers

    @staticmethod
    def deprocess_image(img):
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.1
        img += 0.5
        img = np.clip(img, 0, 1)
        img *= 255
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    @staticmethod
    def preprocess_image(img):
        img -= img.mean()
        img /= img.std() + 1e-5
        img *= 0.1
        img += 0.5
        img = np.clip(img, 0, 1)
        return img

    def save_summary(self):
        with open(os.path.join(self.reports_directory, 'modelsummary.txt'), 'w') as f:
            with redirect_stdout(f):
                self.model.summary()

    def visualization_of_layer_activation(self, image_path):
        # preparing image
        img = image.load_img(image_path, target_size=self.target_size)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255

        img_container = []

        # preparing activation model
        layer_outputs = [layer.output for layer in self.model.layers[:self.number_of_layers]]
        activation_model = models.Model(inputs=self.model.input, outputs=layer_outputs)

        activations = activation_model.predict(img_tensor)
        for layer in range(4):                                                  #ToDo rewrite 4 for all layers and channels
            for channel in range(10):
                fig = plt.figure()
                plt.imshow(activations[layer][0, :, :, channel], cmap='viridis')
                img_container.append(activations[layer][0, :, :, channel])
                plt.axis('off')
                fig.savefig(os.path.join(self.reports_directory, f'layer{layer}_channel{channel}'))
                plt.close()

                #ToDo write separate function to write all reports and fix returning images
        # return img_container

    def visualization_of_max_gradient(self, layer_index=0, filter_index=0):
        # getting gradient from imputes
        layer_output = self.model.layers[layer_index].output
        # layer_output = self.model.get_layer(layer_name).output
        loss = K.mean(layer_output[:, :, :, filter_index])
        grads = K.gradients(loss, self.model.input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        iterate = K.function([self.model.input], [loss, grads])
        loss_value, grads_value = iterate([np.zeros((1, self.target_size[0], self.target_size[1], 3))])
        # stochastic gradient loss loop
        input_img = np.random.random((1, self.target_size[0], self.target_size[1], 3)) * 20 + 128
        step = 1
        for i in range(40):
            loss_value, grads_value = iterate([input_img])
            input_img += grads_value * step

        # tf.compat.v1.enable_eager_execution()
        return self.deprocess_image(input_img[0])

    def accuracy_test(self):

        total_files = 0
        for base, dirs, files in os.walk(self.data_directory):
            for Files in files:
                total_files += 1

        accuracy = 0
        for i in range(total_files):
            fname = os.path.join(self.data_directory, os.listdir(self.data_directory)[i])
            img_cat = image.load_img(fname, target_size=self.target_size)
            x = image.img_to_array(img_cat)
            x = x.reshape((1,) + x.shape)
            x = x / 255
            # prediction
            prediction = self.model.predict(x)
            if prediction < 0.5: #ToDo test accuracy to every model
                accuracy += 1
        accuracy = 100 * accuracy / total_files

        return accuracy

    def heat_map(self, image_path, layer_index):
        img = image.load_img(image_path, target_size=self.target_size)
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor = self.preprocess_image(img_tensor)

        output_vector = self.model.output[:]
        layer_output = self.model.layers[layer_index].output
        print(self.model.layers[layer_index].name)
        grads = K.gradients(output_vector, layer_output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([self.model.input], [pooled_grads, layer_output[0]])

        pooled_grads_value, layer_output_value = iterate([img_tensor])
        for i in range(128):
            layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap
