"""
File splits labeled data into a a training and testing
data set for future evalution of the model. Data points in the
training set is then passed into the defined model which is trained
on a specified epoch value
"""


import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
import tensorflow as tf

actions = np.array(['perfect_shot', 'ok_shot', 'bad_shot', 'unrecognizable'])

no_sequences = 4

sequence_length = 90

DATA_PATH = os.path.join('SHOOTING_DATA')

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

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
print("Size of X:", X.shape)
y = to_categorical(labels).astype(int)
print("Size of y:", y.shape)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Define the model
model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='tanh', input_shape=(sequence_length, X.shape[2])))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, activation='tanh'))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=False, activation='tanh'))
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#reduces learning rate by .2 if validation loss doesnt improve after 10 epochs
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.00001)
model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback, reduce_lr])

model.save('shooting_form_v1.h5')

# Display the model summary
model.summary()