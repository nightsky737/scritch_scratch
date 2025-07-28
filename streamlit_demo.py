'''
Gonna do a demo w/ streamlit

- Let you create a homemade model and overfit to it? Or use it for backprop on simple fxns? 
- Let you train jax and perhaps display metrics?
- let you train the cnn.

'''

import streamlit as st
from basics.model import *
from basics.jaxmodel import *
from CNN.CNN import *
import numpy as np
import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz", )

indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

def batch(x, y, batch_size=64):
    if len(x) % batch_size != 0:
        x = x[:batch_size * (len(x)//batch_size)]
        y=y[:batch_size*(len(x)//batch_size)]
    return np.array(np.split(x, int(len(x) / batch_size), axis=0)), np.split(y, int(len(y)/batch_size), axis=0)

def fix_data(x, y):
    x = x.reshape(x.shape[0], 28*28 )/255
    test = np.zeros((x.shape[0], 10))
    test[np.arange(x.shape[0]),y] = 1
    return (x, test)

fixed_x, fixed_y = fix_data(x_train, y_train)
x_test, y_test = batch(*fix_data(x_test[:1000], y_test[:1000]), 32)
b_x , b_y = batch(fixed_x[:10000], fixed_y[:10000], 64)

b_x_cnn = b_x.reshape(b_x.shape[0], b_x.shape[1], 28, 28, 1)
x_test_cnn = x_test.reshape(x_test.shape[0],  x_test.shape[1], 28, 28, 1)

b_x_flat = b_x.reshape(b_x.shape[0], b_x.shape[1], 28*28)
x_test_flat = x_test.reshape(x_test.shape[0], x_test.shape[1], 28*28)

jaxmodel = JaxModel(28*28, 10, [ 8, 16], jax_mse, jax_sigmoid) 


st.write('Hello')
if st.button('click me'):
    if 'total' not in st.session_state:
        st.session_state.total=0 # or st.session_state['total'] = 0 -> kind of like uh the flask memory thing.
    st.session_state.total += 1
    st.write(st.session_state.total)
    datas = []

    for _epoch in range(20):
        st.write(f"starting epoch {_epoch}")
        st.write(jaxmodel.train_epoch(b_x, b_y,  (x_test, y_test), lr=1e-2 ))
        
    