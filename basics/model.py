import numpy as np
import math
from grad import *
import time

def weight_matrix(shape, naive=False):
    """weight matrix thingy. Give the shape of weight matrix you want"""
    number = 1
    if(type(shape) == int):
        shape = [shape]
    for i in shape:
        number*= i
    if naive:
        return np.array([Number(i / 10) for i in range(number)]).reshape(*shape)
    #Uses xavier init if 2d
    #if len(shape) == 2:
    #    return np.array([Number(np.random.uniform(low=-math.sqrt(6/shape[0] ), high=math.sqrt(6/shape[0]), size=None)) for i in range(number)]).reshape(*shape)
    return np.array([Number(np.random.uniform(low=-.2, high=.2, size=None)) for i in range(number)]).reshape(*shape)

#Activation fxns:
def sigmoid(x):
    return np.vectorize(lambda x: 1/(1+math.e**-x))(x)


def softmax(x):
    # subtract max for numerical stability
    e_x = np.vectorize(lambda x: math.e**x)(x)
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def relu(x):
    def relu_one(k):
        if k <= 0:
            return 1e-2 * k
        else:
            return k
    return np.vectorize(lambda x: relu_one(x))(x)

def topo_sort(N):
    '''Returns the best order to backprop N in. This is such that if you call backprop(N), the gradient on N is final so you don't have to fully traverse the graph every time.
    
    returns a list of Numbers.
    
    So ideally we'd just backprop one layer (to N's two creators (ie a and b when u backprop N in a * b = N) when we get to the N 

    This will do a big stupid if a node appears twice (it will schedule the ones that appear twice to run its backprop when we first encounter them, not taking into account the second time),
    but the only ones that appear twice are weights and biases and those don't have anything after them in the graph that depends on their gradients
    wait could I optimize it to not backprop weights? No it alr does that it doesnt have any creators
    '''
    ret = []
    visited = {}
    stk=[(N, False)]
    while(len(stk) != 0):
        number, post = stk.pop()
        '''
        dfs's a Number, following it down the path by df'sing its two creators.
        '''
        if post:
            ret.append(number)
            continue
            
        if visited.get(number)== "visited":
            continue
        if visited.get(number) == "visiting":
            print("ahahaaha uve gota cycle") #this should NEVER Happen
            continue
            
        visited[number] = "visiting"
        
        if number.creator != None:
            stk.append((number.creator.a, False))
            stk.append((number.creator.b, False))

        stk.append((number, True))
        visited[number] = "visited"
        
    return ret

"""
def old_sort(N):
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
 """
    
class Model():
        
    def __init__(self, input_size, output_size, hidden_layers, naive=False, seed=None, debug=False):
        '''
        Takes list of # of things in their layers.
        Input_size and output_size are the number of input/output neurons
        hidden layers is the sizes of the hidden layers

        naive is only for testing purposes 
        '''
        if seed != None:
            np.random.seed(seed)

        #Hidden layer has dims of all hidden layers.
        hidden_layers.append(output_size)
        self.layer_sizes = hidden_layers
        self.layers = []
        self.biases = []
        self.debug=False

        #Hidden states is after the *weight but before activation fxn. These are mainly for debugging.
        if self.debug:
            self.hidden_states = []
            self.hidden_states_activation = []
        
        prev_size = input_size
        
        for hidden_layer in hidden_layers:
            self.layers.append(weight_matrix([prev_size, hidden_layer], naive))
            self.biases.append(weight_matrix(hidden_layer, naive))
            prev_size  = hidden_layer
            
        hidden_layers.append(input_size)

    def fd(self, x):
        '''f pass with input. '''
        if self.debug:
            self.hidden_states_activation = []
            self.hidden_states = []

        self.input = x
        for i in range(len(self.layers)):
            x = x @ self.layers[i]
            x += self.biases[i]

            if self.debug:
                self.hidden_states.append(x)

            if i != len(self.layers) - 1:
                x = relu(x)
            else:
                x = sigmoid(x) #Only sigmoid the last one.
            if self.debug:
                self.hidden_states_activation.append(x)

        return x
        
    def train_epoch(self, x, y, lr=10**-2, timer=False, batch_timer = True):
        '''
        f pass and then gradient descent.
        x: Input. Not flat as a pancake
        y: the goal. In sparese tensor 
        lr: how quick it learns
        '''
        if timer:
            start_time = time.perf_counter() 
        full_start = time.perf_counter()

        num_correct = 0
        losses = []
        weight_sizes = []

        for i in range(len(y)):
            pred = self.fd(x[i])

            batch_start_time = time.perf_counter()
            if timer:
                print(f"Elapsed time for fd pass: { time.perf_counter()  - start_time} seconds")
                start_time = time.perf_counter() 
           
            y = np.array(y)

            for layer in self.layers:
                 for w in layer.flat:
                     w.null_gradients(recursive=False)
            for bias in self.biases:
                 for b in bias.flat:
                     b.null_gradients(recursive=False)

            if timer:
                print(f"Elapsed time for null gradients: { time.perf_counter()  - start_time} seconds")
                start_time = time.perf_counter() 

            # print(pred.shape) #32, 10
            mse = np.sum(pred * pred - 2 * pred * y + y * y) #/(len(pred) * len(y))
            num_correct += np.sum(np.argmax(pred, axis=1) == np.argmax(y[i], axis=1))
            losses.append(mse.data)

            if timer:
                print(f"Elapsed time for mse stuff: { time.perf_counter()  - start_time} seconds")
                start_time = time.perf_counter() 
            # print("mse", mse)
            mse_sorted = topo_sort(mse)
            # print(f"{len(mse_sorted)}")

            if timer:
                print(f"Elapsed time for topo sorting mse gradients: { time.perf_counter()  - start_time} seconds")
                start_time = time.perf_counter() 
            weight_sizes.append(len(mse_sorted))

             # print(f"order {mse_sorted}")
            for num in mse_sorted:
                num.backprop_single()
            if timer:
                print(f"Elapsed time for backprop: { time.perf_counter()  - start_time} seconds")
                start_time = time.perf_counter() 
        
            # for i, layer in enumerate(self.layers):
            #     print(f"Layer {i} avg grad:", np.mean([w.grad for w in layer.flat]))

            for i, layer in enumerate(self.layers):
                for w in range(len(layer.flat)):
                    layer.flat[w] = Number(layer.flat[w] - layer.flat[w].grad * lr)

            for i in range(len(self.biases)):
                layer = self.biases[i]
                for b in range(len(layer.flat)):
                    layer.flat[b] = Number( layer.flat[b] - layer.flat[b].grad * lr)
            if timer:
                print(f"Elapsed time for gradient updates { time.perf_counter()  - start_time} seconds")
                start_time = time.perf_counter() 
            if batch_timer:
                print(f"Elapsed time for one batch: { time.perf_counter()  - batch_start_time} seconds")
                start_time = time.perf_counter()

        print(f"Acc: {num_correct/(y.shape[1] * len(y))} Avg loss: {sum(losses)/len(y)}")
        print(f"Elapsed time for one epoch: { time.perf_counter()  - full_start} seconds")

        return losses
 
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