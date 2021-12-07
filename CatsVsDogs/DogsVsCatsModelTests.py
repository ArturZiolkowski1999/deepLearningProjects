import os
import random

from NetworkReport import NetworkReport
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from keras.models import load_model
from keras.preprocessing import image
from random import randrange
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16

# preparing directorys
reports_dir = os.path.join(os.getcwd(), 'Report')
models_dir = os.path.join(os.getcwd(), 'Models')
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

# load model
model = load_model(os.path.join(models_dir, 'DogsVsCats_big_2.h5'))


report = NetworkReport(model, reports_dir, models_dir, test_cats_dir, 500, (150, 150))
print(report.number_of_layers)
report_VGG16 = NetworkReport(VGG16(weights='imagenet'), reports_dir, models_dir, test_cats_dir, 500, (224, 224))
print(report_VGG16.number_of_layers)

# report.save_summary()
# report.visualization_of_layer_activation(os.path.join(test_cats_dir, os.listdir(test_cats_dir)[9]))
# print(report.accuracy_test())
# imag = report.visualization_of_max_gradient(2, 5)
# plt.imshow(imag)
# plt.show()


report_VGG16.save_summary()
heatmap = report_VGG16.heat_map(os.path.join(test_cats_dir, os.listdir(test_cats_dir)[0]), 17)
plt.imshow(heatmap)
plt.show()
# heatmap = report.heat_map(os.path.join(test_cats_dir, os.listdir(test_cats_dir)[3]), 6, 1)
# plt.imshow(heatmap)
# plt.show()

# img = cv2.imread(os.path.join(test_cats_dir, os.listdir(test_cats_dir)[9]))
# heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# heatmap = np.uint8(255 * heatmap)
#
# heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
# superimposed_img = heatmap * 0.4 + img
# cv2.imwrite(os.path.join(reports_dir, 'heat_map'), superimposed_img)
