
import jax.numpy as jnp
from jax import grad
import jax
import numpy as np
import math

def jax_weight_matrix(shape, naive=False):
    """weight matrix thingy.give dims. Not 0."""
    number = 1
    if(type(shape) == int):
        shape = [shape]
    for i in shape:
        number*= i
    if naive:
        return jnp.array([(i / 10) for i in range(number)]).reshape(*shape)
    return np.array([np.random.uniform(low=-.2, high=.2, size=None) for i in range(number)]).reshape(*shape)
    # return np.array([variable(np.random.uniform(low=-.2, high=.2, size=None)) for i in range(sizes[0] * sizes[1])).reshape(*shape)

def jax_relu(x):
    return jnp.where(x <= 0, 1e-2 * x, x)

def jax_sigmoid(x):
    return jnp.vectorize(lambda x: 1/(1+jnp.e**-x))(x)

def jax_mse(x, y):
    return jnp.sum(x * x - 2 * x * y + y * y)/len(x) #, axis=1 if u wanna mean it 

def jax_softmax(x):
    exponentiated = jnp.exp(x - jnp.max(x, axis=-1, keepdims=True)) #Subtract the max in order to keep it from overflowing silently 
    return exponentiated / jnp.sum(exponentiated, axis=-1, keepdims=True) #oh wow keepdims is possible.
    
def jax_cross_entropy(x, y):
    return -jnp.sum(y * jnp.log(x)) / len(y)

class JaxModel():
    def __init__(self, input_size, output_size, hidden_layers, loss_fn=jax_mse, activation_fn=jax_sigmoid, naive=False, seed=None):
        '''
        Honestly jax doesnt play great with class structures but thats fine.
        '''
        if seed != None:
            np.random.seed(seed)
        self.layer_sizes = hidden_layers
        self.layers = []
        self.biases = []
        self.loss_fn = loss_fn
        self.activation_fn = activation_fn
        self.eval_activation_fn = activation_fn

        # if loss_fn == jax_cross_entropy:
        #     self.activation_fn = None

        prev_size = input_size
        
        for hidden_layer in hidden_layers:
            self.layers.append(jax_weight_matrix([prev_size, hidden_layer], naive))
            self.biases.append(jax_weight_matrix(hidden_layer, naive))
            prev_size  = hidden_layer
            
        self.biases.append(jax_weight_matrix([output_size]))
        self.layers.append(jax_weight_matrix([prev_size, output_size]))

        self.layers= tuple(self.layers)
        self.biases = tuple(self.biases)  


    def fd(self, x):
        '''f pass with input. '''
        for i in range(len(self.layers)):
            x = x @ self.layers[i]
            x += self.biases[i]
            if i != len(self.layers) - 1:
                x = jax_relu(x)
            else:
                x = self.eval_activation_fn(x)
        return x
    
    def loss_static(self, params, x, y):
        '''f pass with for loss.  '''
        w, b = params
        for i in range(len(b)):
            x = x @ w[i]
            x += b[i]
            if i != len(b) - 1:
                x = jax_relu(x)
            elif self.activation_fn is not None:
                x = self.activation_fn(x)
        y = jnp.array(y)
        return self.loss_fn(x, y) #Now it just passes it to the loss fn. NO ifs cause jax doesnt like if

            
    def train_epoch(self, x, y, test_data, lr=10**-2): 
        '''
        f pass and then uh gradient descent?

        x:  
        y: the goal. In not sparse tensor.
        lr: how quick it learns
        '''
        losses = []
        x = np.array(x)
        
        for batch_num in range(len(y)):
            loss, grads = jax.value_and_grad(self.loss_static, argnums=(0))((self.layers, self.biases), x[batch_num], y[batch_num])
            
            losses.append(loss)
            
            #0 contains weights and 1 contains the bias grads. 
            # print([i.shape for i in grads[1]])
            # print([i.shape for i in grads[0]])

            self.layers = list(self.layers)
            self.biases = list(self.biases)
            

            for i, (layer, grad_layer) in enumerate(zip(self.layers, grads[0])):
                self.layers[i] = layer - lr * grad_layer  
                
            for i, (bias, grad_bias) in enumerate(zip(self.biases, grads[1])):
                self.biases[i] = bias - lr * grad_bias   

        #lmao we just do this on the last of test data not even running acc or anything
        testX, testY = test_data
        preds = self.fd(testX[0]) 
        # print(preds)
        correct = jnp.sum(jnp.argmax(preds, axis=1) == jnp.argmax(testY[0], axis=1))
        acc = correct / len(testY[0])
        
        avg_loss = jnp.mean(jnp.array(losses))
        print(f"Loss: {avg_loss:.4f}")
        return losses, (acc, avg_loss)