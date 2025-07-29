'''
Gonna do a demo w/ streamlit

- Let you create a homemade model and overfit to it? Or use it for backprop on simple fxns? 
- Let you train jax and perhaps display metrics?
- let you train the cnn.

'''

import streamlit as st
# from basics.model import *
from basics.jaxmodel import *
from basics.model import *
import numpy as np
import keras
import matplotlib.pyplot as plt
import random

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

b_x_flat = b_x.reshape(b_x.shape[0], b_x.shape[1], 28*28)
x_test_flat = x_test.reshape(x_test.shape[0], x_test.shape[1], 28*28)

single_bx, single_by = batch(fixed_x[:1], fixed_y[:1], 1)

session_state = st.session_state

st.write('Welcome to the demo of the feed forward model and gradient descent!')

if st.button("Click to get an image to fit with gradient descent! (Rerun this if pre-fit prediction is same as truth)"):
    img = single_bx[0][0]
    truth = single_by[0]
    st.session_state['bx'] = single_bx
    st.session_state['by'] = single_by
    st.session_state['crapmodel'] =  Model(28*28, 10, [ 8, 16])
    fig, ax = plt.subplots()
    ax.axis('off')
    predicted = st.session_state.crapmodel.fd(np.array(img.flat))

    st.session_state['prefit'] = np.argmax(predicted)
    ax.set_title(f"Truth:{np.argmax(truth)} Pre-fit prediction:{np.argmax(predicted)}")
    ax.imshow(img.reshape(28, 28))
    st.pyplot(fig)

if st.button("Click to fit the image with gradient descent"):
    if st.session_state.get('crapmodel') == None: 
        st.write("Get the image first!")
    else:
        img = st.session_state['bx'][0][0]
        truth = st.session_state['by'][0]

        for _epoch in range(10):
            losses = st.session_state.crapmodel.train_epoch(st.session_state['bx'], st.session_state['by'], lr=1, timer=False, batch_timer=False)
            st.write(f"Average loss for epoch {_epoch}: {sum(losses)/len(losses)}")
        fig, ax = plt.subplots()

        ax.axis('off')
        predicted = st.session_state.crapmodel.fd(np.array(img.flat))
        ax.set_title(f"Truth:{np.argmax(truth)} Pre-fit prediction: {st.session_state['prefit']} Post-fit prediction:{np.argmax(predicted)}")
        ax.imshow(img.reshape(28, 28))
        plt.show()
        st.pyplot(fig)

st.write('Now for the feed forward model!')
st.write("first, set some hyperparameters. A known working combination that gets relatively quick results is lr=.1, softmax, and cross entropy for 5-7 epochs")
lr= st.number_input("Learning rate")
activation_fn=st.selectbox("Activation Function", options=["softmax", "sigmoid"])
loss_fn=st.selectbox("Loss Function", options=["mse", "cross entropy"])
num_epochs = st.number_input("num epochs to train", step=1)

if activation_fn == "softmax":
    activation_fn = jax_softmax
else:
    activation_fn = jax_sigmoid

if loss_fn == "mse":
    loss_fn = jax_mse
else:
    loss_fn = jax_cross_entropy

if st.button("Click to start training the model. (this can take a few mins)"):
    st.session_state.jaxmodel = JaxModel(28*28, 10, [ 8, 16], loss_fn, activation_fn) 

    datas = []
    for _epoch in range(num_epochs):
        losses, data = st.session_state.jaxmodel.train_epoch(b_x, b_y,  (x_test, y_test), lr=lr)
        acc, avg_loss = data
        datas.append(f"Epoch: {_epoch} Acc: {acc * 100:.4f}% Loss: {avg_loss:.4f}")
        st.write(f"Epoch: {_epoch} Acc: {acc * 100:.4f}% Loss: {avg_loss:.4f}") 
    session_state['jaxdata'] = datas

# if "jaxdata" in session_state:
#     for log in session_state["jaxdata"]:
#         st.write(log)

if st.button("Click to show some of the model's predictions!"):
    fig = plt.figure(figsize=(10, 7))
    pic = 1
    for i, img in enumerate(x_test[0][:10]):
        plt.subplot(2, 5, pic)
        plt.axis('off')
        predicted = st.session_state.jaxmodel.fd(jnp.array(img.flat))
        plt.title(f"Truth:{np.argmax(y_test[0][i])} Predicted:{jnp.argmax(predicted)}")
        plt.imshow(img.reshape(28, 28))
        pic+= 1
        plt.show()
    st.pyplot(fig)