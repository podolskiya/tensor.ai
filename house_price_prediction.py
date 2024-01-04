# This is a simple project just to experiment and solidify my foundamental understanding of tensor.
# import libraries
import tensorflow as tf
import numpy as np
from tensorflow import keras

def house_model():
  # Define input and output tensors with the values for houses with 1 up to 6 bedrooms in a linear relationship where house
  # costs 50k and it costs extra 50k per bedroom - 1 bedroom 100k, 2 bedroom 150k)
  xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
  ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
  
  # Define model with 1 dense layer and 1 unit
  model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
  
  # Compile model with Stochastic Gradient Descent as optimizer and Mean Squared Error as the loss function
  model.compile(optimizer='sgd', loss='mean_squared_error')

  # Train your model for 1000 epochs
  model.fit(xs, ys, epochs=1000)
  return model

# Get the trained model
model = house_model()

# Predict - should be close to 4
new_x = 7.0
prediction = model.predict([new_x])[0]
print(prediction)
  
