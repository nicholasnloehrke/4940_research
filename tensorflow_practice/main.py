import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf


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
#   training:   70%
#   test:       15%
#   validation: 15%
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# sequential NN of dense (fully connected) layers
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, input_dim=X_train.shape[1], activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), batch_size=32)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f'Accuracy: {test_accuracy:.4f}')

print(model.summary())