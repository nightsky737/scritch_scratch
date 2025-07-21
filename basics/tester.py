import tensorflow as tf
import keras
from grad import *
from model import *
import numpy as np

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

fixed_x, fixed_y = fix_data(x_train[:1000], y_train[:1000])
b_x , b_y = batch(fixed_x, fixed_y, 32)

test_bx = [b_x[0] for i in range(3)]
test_by = [b_y[0] for i in range(3)] #Fun challenge: Can you overfit to this?

my_model = Model(28*28, 10, [4, 8])
datas = []
for _epoch in range(10):
    print(f"starting epoch {_epoch}")
    datas.append(my_model.train_epoch(test_bx, test_by, lr=1e-3, timer=False, batch_timer=False))

