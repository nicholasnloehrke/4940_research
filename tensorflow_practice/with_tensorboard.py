import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import os

# Load the dataset
data = pd.read_csv("student_data.csv", delimiter=';')

# map output to number
#   Dropout  -> 0
#   Enrolled -> 1
#   Graduate -> 1
data['Output'] = data['Output'].apply(lambda x: 0 if x == 'Dropout' else 1)

# X is our input vector (excludes the 'Output' column)
X = data.drop(columns=['Output'])

# y is our output vector
y = data['Output']

# normalize the input vector
X_scaled = StandardScaler().fit_transform(X)

# split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# define the log directory for TensorBoard
log_dir = os.path.join("logs", "fit", "model_1")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# sequential NN of dense (fully connected) layers
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_dim=X_train.shape[1], activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the model with TensorBoard callback
model.fit(X_train, y_train, epochs=25, batch_size=8, callbacks=[tensorboard_callback])

# evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {test_accuracy:.4f}')

print(model.summary())
