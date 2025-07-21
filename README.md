# READ THIS
This is honestly pretty useless to anyone except me. This was a project I've created to just understand deep learning better, and while there are features that work, I didn't prioritize making them accessible via a neat api or website. That being said, there are still a few runnable notebooks

# Instructions for use

1. Clone the repo.
2. Inside will be a couple folders. Basics holds the model basics (ie the autodiff library and a simple ffnn). There are 2 versions implemented there, one from scratch and one using Jax. Their demo code are in, respectively, autodiff_check.ipnyb, and jaxmodel.ipynb.
3. Further instructions might come up as you run the code in the notebooks.

# My notes:
Neural nets and models have always just been a black box to me, and well, it's still a black box but it's now RGB(1, 1, 1) instead of RGB(0, 0, 0)

I also did part of this project prior to shipwrecked, having written most of the base backprop from scratch library as part of another program, and then extending it with the creation of the model based on my backprop and a jax model from nearish scratch for reference and learning more about the loss and activations fxns. I was about halfway through that when shipwrecked started, and thus have logged 15 more hours finishing the jax and homemade ffnns, and []

# Other notes:
There isn't much actual data pipelining (ie val, shuffle, data cleaning, etc, this is more just focused on the model itself.)