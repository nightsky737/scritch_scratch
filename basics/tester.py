import tensorflow as tf
import keras
from grad import *
from model import *
import numpy as np
import jax.numpy as jnp
import jax

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz", )
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

def batch(x, y, batch_size=32):
    if len(x) % batch_size != 0:
        x = x[:batch_size * (len(x)//batch_size)]
        y=y[:batch_size*(len(x)//batch_size)]
    return np.array_split(x, len(x) / batch_size, axis=0), np.array_split(y, len(y)/batch_size, axis=0)

def fix_data(x, y):
    x = x.reshape(x.shape[0], 28*28)/255
    test = np.zeros((x.shape[0], 10))
    test[np.arange(x.shape[0]),y] = 1
    return (x, test)

fixed_x, fixed_y = fix_data(x_train[:1], y_train[:1])
b_x , b_y = batch(fixed_x, fixed_y, 1)

my_model = Model(28*28, 10, [ 8, 16])
datas = []
for _epoch in range(100):
    print(f"starting epoch {_epoch}")
    datas.append(my_model.train_epoch(b_x, b_y, lr=1e-2, timer=False, batch_timer=False))

