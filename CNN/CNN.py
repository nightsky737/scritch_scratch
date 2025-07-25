
import jax.numpy as jnp
from jax import debug
from jax import grad
import jax
import math
import matplotlib.pyplot as plt
import optax
from PIL import Image
import numpy as np

def make_kernel(shape,input_c, output_c, key, naive=False):
    """weight matrix thingy. Returns (shape, shape, input_c, output_c)"""
    #my gradients are either going poof or kaboom so here goes nothing:
    scale = jnp.sqrt(2.0 / (shape * shape * input_c + shape * shape * output_c))  # He init
    return jax.random.normal(key, (shape,shape,input_c, output_c)) *scale


def make_biases(num_filters,key, naive=False):
    """weight matrix thingy.give dims. Not 0."""
    # if naive:
    #     return jnp.array([(i / 10) for i in range(number)]).reshape(*shape)
    
    return jax.random.normal(key, (num_filters)) * 0.01

    
def sigmoid(x):
    return jnp.vectorize(lambda x: 1/(1+math.e**-x))(x)


def relu(x):
    return jnp.where(x <= 0, 1e-2 * x, x)

def convolve(x, filters):
    '''
    Filters in shape: (k_size, k_size,  c_in, c_outs)
    x is input in BHWC
    '''
    
    offset = filters.shape[1]//2 
    # print(filters.shape)
    # print(x.shape)
    #Get the shape of the returned thing.
    ret_b, ret_h, ret_w, ret_f = (x.shape[0], x.shape[1] - 2 * offset, x.shape[2] - 2 * offset, filters.shape[3])
    ret = jnp.zeros((ret_b, ret_h, ret_w, ret_f))
    

    for i in range(offset,ret_h):
        for j in range(offset, ret_w):
            area = x[:,i - offset:i+offset + 1, j - offset: j + offset + 1,:]
            to_set = jnp.einsum("bhwc,hwcf->bf", area, filters) #here h and w are the sizes of the conv kernel not the actual img
            ret = ret.at[:,i-offset,j-offset,:].set(to_set)

    return ret

 
@jax.jit 
def loss_static(weights, x, y):
    '''f pass with for loss.  Weights is tuple of [kernels, bias, ffnn]'''
    #fd pass. 
    k, b, ffnn = weights
    for i in range(len(k)): 
        # print(f"Shape of kernels in this layer {kernels[i].shape} ")
        #jax debug basically has jax print the stuff so its not like traced tensors

        # debug.print("Activations- Max: {max:.5f}, Min: {min:.5f}, Avg: {mean:.5f}", 
        #         max=jnp.max(x),min=jnp.min(x),  mean=jnp.mean(x))

        x =  convolve( x,k[i]) 
        x += b[i][None, None, None, :]
        x = relu(x)
    x = jnp.reshape(x, shape=(x.shape[0], -1))
    # print(ffnn.shape)
    # print(x.shape)
    x = x @ ffnn #logitssss
    y = jnp.array(y)
    return optax.softmax_cross_entropy(x, y).mean() #softmax_ce takes logits as the arg.

# @jax.jit
def fd(weights, x):
    k, b, ffnn = weights
    for i in range(len(k)):
        x = convolve(x, k[i])
        x += b[i][None, None, None, :]
        x = relu(x)
    x = jnp.reshape(x, shape=(x.shape[0], -1))

    x = x @ ffnn #logitssss again
    return x 

class Model():
    def __init__(self, params, naive=False, seed=0):
        '''
        Takes list of # of things in their layers.
        Layers are outputs?
        params has:
            kernels_info is a tuple with pairs of (kernel size, num out filters) 
            input_shape (hwc)
            output shape
        '''
        key = jax.random.key(seed)
 
        kernels = []
        biases = [] 

        in_h, in_w, in_c = params["input_shape"]
        num_prev_filters = in_c

        for(kernel_size, num_filters) in params["kernel_info"]:
            key, b_key, k_key = jax.random.split(key, num=3)

            kernels.append(make_kernel(kernel_size, num_prev_filters, num_filters, k_key)) 
            biases.append(make_biases(num_filters, b_key))

            in_h -= kernel_size -1
            in_w -= kernel_size - 1
            num_prev_filters = num_filters
        
        key, other = jax.random.split(key, num=2)
        scale = jnp.sqrt(2.0 / (num_prev_filters * in_h * in_w + params["output_size"]))
        self.ffnn = jax.random.normal(key, (num_prev_filters * in_h * in_w, params["output_size"])) * scale
        self.kernels = kernels
        self.biases = biases
        self.params = params
    
    
    def print_layer_info(self):
        print("printing layer info")
        for i in range(len(self.params["kernel_info"])):
            print(f"layer {i} kernel shape {self.kernels[i].shape}")
            print("Kernels in layer"  + str(i) + 
                  f" has max of {jnp.max(self.kernels[i])} min of {jnp.min(self.kernels[i])} avg of {jnp.mean(self.kernels[i])}" )
            print(f"layer {i} bias shape {self.biases[i].shape}")
        print(f"ffnn shape {self.ffnn.shape}")

    def convolve(self, x, filters):
        return convolve(x, filters)

    def fd(self, x):
        weights = (self.kernels, self.biases, self.ffnn)
        return jax.nn.softmax(fd(weights, x))
    
            
    def train_epoch(self, x, y, test_data, lr=10**-2):
        '''
        f pass and then gradient descent
        '''
        losses = []
        running_correct = 0
        for i in range(len(y)):
            loss, grads = jax.value_and_grad(loss_static, argnums=(0))(
                (self.kernels, self.biases, self.ffnn), x[i], y[i]
            )
            losses.append(loss)
            
            for w_num in range(len(self.kernels)):
                self.kernels[w_num] = self.kernels[w_num] - lr * grads[0][w_num]

            for w_num in range(len(self.biases)):
                self.biases[w_num] = self.biases[w_num] - lr * grads[1][w_num]

            self.ffnn = self.ffnn - lr * grads[2]

            preds = self.fd(x[i]) 
            running_correct += jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(y[i], axis=1))


        testX, testY = test_data
        preds = self.fd(testX[0]) 
        correct = jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(testY[0], axis=1))
        acc = correct / len(testY[0])
        # acc = running_correct / (len(y[0]) * len(y))
        avg_loss = jnp.mean(jnp.array(losses))
        print(f"Acc: {acc:.4f} Loss: {avg_loss:.4f}")
        return losses
        return losses