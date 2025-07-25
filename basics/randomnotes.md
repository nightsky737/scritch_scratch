# Model notes
Cross entropy loss could be used w/ softmax and sigmoid
mse can work with softmax/sigmoid but typically not as well as cross entropy (as the squishing due to softmax can result in weirdness w/ squaring.)

softmax and sigmoid are different. both use the exponents but in very different ways.

Jax overflows are often silent
Jax is always going to be better than yours. 

# General coding notes
stick to it (sometimes first few epochs were crap but it figured it out later on)
Be explicit with fxns (jax sigmoid versus regular sigmoid). dont be lazy. TYpe out the extra jax.


# To learn
What is the log sum exp trick?