"""
File splits labeled data into a a training and testing
data set for future evalution of the model. Data points in the
training set is then passed into the defined model which is trained
on a specified epoch value
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

actions = np.array(['perfect_shot', 'ok_shot', 'bad_shot', 'unrecognizable'])

# Load the saved test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.dot(x, self.W) + self.b
        e = tf.keras.backend.tanh(e)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

model = load_model('shooting_form_v1.h5', custom_objects={'AttentionLayer': AttentionLayer})

y_pred = model.predict(X_test)

y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("Unique labels in y_test_labels:", np.unique(y_test_labels))
print("Unique labels in y_pred_labels:", np.unique(y_pred_labels))

# Evaluate 
accuracy = accuracy_score(y_test_labels, y_pred_labels)
classification_rep = classification_report(y_test_labels, y_pred_labels, target_names=actions, labels=[0, 1, 2, 3])
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Save evaluation metrics 
with open('evaluation_metrics_v1.txt', 'w') as file:
    file.write(f"Test Accuracy: {accuracy}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_rep + '\n\n')
    file.write("Confusion Matrix:\n")
    file.write(str(conf_matrix))

print(f"Test Accuracy: {accuracy}")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)
