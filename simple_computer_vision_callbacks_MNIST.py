import os
import tensorflow as tf
from tensorflow import keras

current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
data_path = os.path.join(current_dir, "data/mnist.npz")
        
# Normalize pixel values
x_train = x_train / 255.0

data_shape = x_train.shape

print(f"Number of examples: {data_shape[0]}. Shape: ({data_shape[1]}, {data_shape[2]})")

# Callback at accuracy > 99%
class myCallback(tf.keras.callbacks.Callback):
        # Define the correct function signature for on_epoch_end
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.99):                 
                print("\nReached 99% accuracy so cancelling training!")
                
                # Stop training once the above condition is met
                self.model.stop_training = True

# Create model

def train_mnist(x_train, y_train):
    callbacks = myCallback()
    model = tf.keras.models.Sequential([         
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        
        
    ]) 
    model.compile(optimizer=tf.optimizers.Adam(),                   
                  loss='sparse_categorical_crossentropy',                   
                  metrics=['accuracy'])     
    
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])


    return history

hist = train_mnist(x_train, y_train)
