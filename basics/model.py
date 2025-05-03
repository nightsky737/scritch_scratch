import numpy as np
import math
from grad import *

def weight_matrix(shape, naive=False):
        """weight matrix thingy.give dims. Not 0."""
        number = 1
        if(type(shape) == int):
            shape = [shape]
        for i in shape:
            number*= i
        if naive:
            return np.array([Number(i / 10) for i in range(number)]).reshape(*shape)
        return np.array([Number(np.random.uniform(low=-.2, high=.2, size=None)) for i in range(number)]).reshape(*shape)
    
def sigmoid(x):
    return np.vectorize(lambda x: 1/(1+math.e**-x))(x)

def relu(x):
    # print(x)
    def relu_one(k):
        if k <= 0:
            return 10e-3 * k
        else:
            return k
    return np.vectorize(lambda x: relu_one(x))(x)

def softmax(x):
    '''uh what is x? That's a good question.

    X will be the shape (batch_size, output_size) output. We then want to run softmax over every entry in the 0 axis. 
    '''

def topo_sort(N):
    '''Returns the correct order to backprop N in. This is such that if you call backprop(N), the gradient on N is final.
    returns a list of Numbers.
    
    So ideally we'd just backprop one layer (to N's two creators (ie a and b when u backprop N in a * b = N) when we get to the N 

    *sigh* back to the usaco grind
    '''
    ret = []
    visited = {}
    def dfs(number, debug=True):
        # print(number)
        '''
        dfs's a Number, following it down the path by df'sing its two creators.
        '''
        if visited.get(number)== "visited":
            return True
        if visited.get(number) == "visiting":
            return False
            
        visited[number] = "visiting"
        if number.creator != None:
            if not dfs(number.creator.a) and debug:
                print("please pleas please do not ever print this")
            if not dfs(number.creator.b) and debug:
                print("please pleas please do not ever print this")
        visited[number] = "visited"
        ret.append(number)
        return True
        
    dfs(N)
    ret.reverse()
    return ret
        
class Model():
        
    def __init__(self, input_size, output_size, hidden_layers, naive=False, seed=None):
        '''
        Takes list of # of things in their layers.
        Layers are outputs?
        '''
        if seed != None:
            np.random.seed(seed)

        self.layer_sizes = hidden_layers
        self.layers = []
        self.biases = []

        #Hidden states is after the *weight but before activation. These are mainly for debugging.
        self.hidden_states = []
        self.hidden_states_activation = []
        
        prev_size = input_size
        
        for hidden_layer in hidden_layers:
            self.layers.append(weight_matrix([prev_size, hidden_layer], naive))
            self.biases.append(weight_matrix(hidden_layer, naive))
            prev_size  = hidden_layer
            
        self.biases.append(weight_matrix([output_size]))
        self.layers.append(weight_matrix([prev_size, output_size]))
  
    def fd(self, x):
        '''f pass with input. Input has to be flat like a pancake'''
            
        self.hidden_states_activation = []
        self.hidden_states = []
        self.input = x
        for i in range(len(self.layers)):
            # print(np.max(x))
            x = x @ self.layers[i]
            x += self.biases[i]
            self.hidden_states.append(x)
            if i != len(self.layers) - 1:
                x = relu(x)
            else:
                x = sigmoid(x)
            self.hidden_states_activation.append(x)

        return x
        
    def train_epoch(self, x, y, lr=10**-2):
        '''
        f pass and then uh gradient descent?

        x: Input. Again, flat as a pancake. 
        y: the goal. In sparese tensor. 
        lr: how quick it learns
        '''
        num_correct = 0
        losses = []
        for i in range(len(y)):
                    
            pred = self.fd(x[i])
            # print("preds")
            # print(pred.shape)
            y = np.array(y)
            #wait what am i doing feeding logits in here edit: nvm they're not logits.

            #but we do have a massive issue w/ mse calculations.....
            #RIP to my computational graph idk what happened here but the backprop is gone.
            mse = np.sum(pred * pred - 2 * pred * y + y * y)/(len(pred) * len(y))

            num_correct += np.argmax(pred, axis=1) == np.argmax(y[i], axis=1)
            losses.append(mse)
            print("mse", mse)
            mse_sorted = topo_sort(mse)
            print(f"order {mse_sorted}")

            print(f"sanity check on x {pred[0][0].creator.a}") #oh jeez this is not good. a.creator is None
            for num in mse_sorted:
                num.backprop_single()
                
            # for layer in self.layers:
            #     for w in layer.flat:
            #         if w.grad == None:
            #             print('ur kidding')

            # for bias in self.biases:
            #     for b in bias.flat:
            #         b.null_gradients(recursive=False)

            
            for i, layer in enumerate(self.layers):
                for w in range(len(layer.flat)):
                    layer.flat[w] = Number(layer.flat[w] - layer.flat[w].grad * lr)

            for i in range(len(self.biases)):
                layer = self.biases[i]
                for b in range(len(layer.flat)):
                    layer.flat[b] = Number( layer.flat[b] - layer.flat[b].grad * lr)
                    
        print(f"Acc: {num_correct/len(y)} Avg loss: {sum(losses)/len(y)}")
 
    def print_info(self, verbose=True):
        print("layers " )
        for i in range(len(self.layers)):
            print( f"weight {i} of shape {self.layers[i].shape}")
            print(self.layers[i])
            print("grads")
            print(np.vectorize(lambda x : x.grad)(self.layers[i]))
            
        print("biases ")
        for i in range(len(self.biases)):
            print( f"bias {i} of shape {self.biases[i].shape}")
            print(self.biases[i])
            print("grads")
            print(np.vectorize(lambda x : x.grad)(self.biases[i]))