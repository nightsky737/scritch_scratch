import streamlit as st

st.write("Minimal demo for the feed forward neural network and the Convolutional Neural Network (CNN)")

if st.button("Click to try out the feed forward neural network"):
    st.switch_page("jax_demo.py") 

if st.button("Click to try out the CNN"):
    st.switch_page("cnn_demo.py") # Redirect to 'home.py' page
