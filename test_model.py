import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Actions that we try to detect
actions = np.array(['perfect_shot', 'ok_shot', 'bad_shot', 'unrecognizable'])

# Load the saved test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

# Load the saved model
model = load_model('shooting_form_v3.h5')

# Make predictions
y_pred = model.predict(X_test)

# Convert predictions and true values from one-hot encoded format to label format
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Check the unique values in y_test_labels and y_pred_labels
print("Unique labels in y_test_labels:", np.unique(y_test_labels))
print("Unique labels in y_pred_labels:", np.unique(y_pred_labels))

# Evaluate the model
accuracy = accuracy_score(y_test_labels, y_pred_labels)
classification_rep = classification_report(y_test_labels, y_pred_labels, target_names=actions, labels=[0, 1, 2])
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels)

# Save evaluation metrics to a text file
with open('evaluation_metrics_v1.txt', 'w') as file:
    file.write(f"Test Accuracy: {accuracy}\n\n")
    file.write("Classification Report:\n")
    file.write(classification_rep + '\n\n')
    file.write("Confusion Matrix:\n")
    file.write(str(conf_matrix))
