import os
from numpy import shape
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
# from tensorflow.keras.datasets import mnist

physical_devices=tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

mnist = keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train=x_train.reshape(-1,784).astype("float32") / 255.0
x_test=x_test.reshape(-1,784).astype("float32") / 255.0

# x_train = tf.convert_to_tensor(x_train) automatically done

# Sequential API (very convenient, not very flexible)

# model = keras.Sequential(
#     [
#         keras.Input(shape=(28*28)),
#         layers.Dense(512, activation='relu'),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(10),
#     ]
# )

# model = keras.Sequential()
# model.add(keras.Input(shape=(784)))
# model.add(layers.Dense(512,activation='relu'))
# model.add(layers.Dense(256,activation='relu'))
# model.add(layers.Dense(10))
# print(model.summary()) to debug

#Functional API (A bit more flwxible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512,activation='relu')(inputs)
x = layers.Dense(256,activation='relu')(x)
outputs = layers.Dense(10,activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)