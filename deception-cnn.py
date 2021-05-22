import tensorflow as tf
import numpy as np
from itertools import cycle

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

 
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        '',  # This is the source directory for training images
        classes = ['Deceptive', 'Not Deceptive'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=25,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '',  # This is the source directory for training images
        classes = ['Deceptive', 'Not Deceptive'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=50,
        # Use binary labels
        class_mode='binary',
        shuffle=True)

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (200,200,3)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model.summary()

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
      steps_per_epoch=8,  
      epochs=16,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=3)

model.evaluate(validation_generator)

validation_generator.reset()
preds = model.predict(validation_generator,
                      verbose=1)

fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Results')
plt.legend(loc="lower right")
plt.show()