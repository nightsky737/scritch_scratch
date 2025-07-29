'''
Gonna do a demo w/ streamlit

- Let you create a homemade model and overfit to it? Or use it for backprop on simple fxns? 
- Let you train jax and perhaps display metrics?
- let you train the cnn.

'''
import sys
sys.path.append("..")  # or absolute path

import streamlit as st
# from basics.model import * 
from CNN.CNN import *
import numpy as np
import keras
import matplotlib.pyplot as plt


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


session_state = st.session_state


# st.session_state.jaxmodel = JaxModel(28*28, 10, [ 8, 16], jax_cross_entropy, jax_softmax) 


st.write('Welcome to the very minimal but functional demo of the Convolutional Neural Net!')
st.write("first, set some hyperparameters. A known working combination that gets relatively quick results is lr=.01, kernel sizes='5,3,3' and filters per kernel = '4,8,8' for 3 epochs")
lr= st.number_input("Learning rate")
kernel_sizes = st.text_input("Kernel sizes, separated by commas. For example, '5,3' would make a kernel of size 5 followed by one of size 3 (do not include the quotations in your input)")
kernel_filters = st.text_input("Number of filters per kernel, separated by commas. For example, '4,8' would make the first kernel have 4 filters and the second have 8 (do not include the quotations in your input)")

num_epochs = st.number_input("num epochs to train", step=1)

try:
    kernel_sizes = [int(size) for size in kernel_sizes.split(",")]
except ValueError as e:
    st.write("Invalid input for kernel sizes")
try:
    kernel_filters = [int(filter) for filter in kernel_filters.split(",")]
except:
    st.write("invalid input for filter sizes")
params = {"kernel_info" : list(zip(kernel_sizes, kernel_filters)), "input_shape": (28, 28, 1), "output_size": 10}

if st.button("Click to start training the model. (this can take 10-15 mins to run)"):
    if len(kernel_sizes) != len(kernel_filters):
        st.write("The number of kernel sizes should be the same as the number of filters per kernel")
    else:
        st.write("Starting Training!")
        st.session_state.convmodel = ConvModel(params)
        datas = []
        for _epoch in range(num_epochs):
            losses, data = st.session_state.convmodel.train_epoch(b_x_cnn, b_y,  (x_test_cnn, y_test), lr=lr)
            acc, avg_loss = data
            datas.append(f"Epoch: {_epoch} Acc: {acc * 100:.2f}% Loss: {avg_loss:.4f}")
            st.write(f"Epoch: {_epoch} Acc: {acc * 100:.2f}% Loss: {avg_loss:.4f}") 

if st.button("Click to show some of the model's predictions (this can also take a minute or two)!"):
    fig = plt.figure(figsize=(10, 7))
    pic = 1
    for i, img in enumerate(x_test_cnn[2][:10]):
        plt.subplot(2, 5, pic)
        plt.axis('off')
        predicted = session_state.convmodel.fd(jnp.expand_dims(img, 0))
        plt.title(f"Truth:{np.argmax(y_test[2][i])} Predicted:{jnp.argmax(predicted)}")
        plt.imshow(img.reshape(28, 28))
        pic+= 1
        plt.show()
    st.pyplot(fig)