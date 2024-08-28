import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Layer
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
import tensorflow as tf

actions = np.array(['perfect_shot', 'ok_shot', 'bad_shot', 'unrecognizable'])

no_sequences = 4
sequence_length = 90

DATA_PATH = os.path.join('SHOOTING_DATA')

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

# Load data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        sequence_missing = False  # Flag to track missing files
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            if not os.path.exists(file_path):
                sequence_missing = True
                break
            res = np.load(file_path)
            window.append(res)
        if not sequence_missing:
            sequences.append(window)
            labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=labels)

np.save('X_test_v2.npy', X_test)
np.save('y_test_v2.npy', y_test)

# TensorBoard callback for logging
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Attention Layer
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

# Define the model
def build_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='tanh'), input_shape=(sequence_length, X.shape[2])))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(256, return_sequences=True, activation='tanh')))
    model.add(Dropout(0.3))
    model.add(AttentionLayer())
    model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(actions.shape[0], activation='softmax'))
    
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

model = build_model()

# Model summary
model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)

# Train the model
model.fit(X_train, y_train, epochs=25, validation_split=0.2, callbacks=[tb_callback, reduce_lr])

# Save the model
model.save('shooting_form_v3.h5')
