''''
Semi-devlog:
Started by doing things on paper, running through a few test cases to see if I could generalize
Started with one layer before moving on to two; the key insight was seing the output of one layer as the input as the next and letting me think about it as blocks
Once I got the hang of that, I tried implementing
Ran into issues with biases shapes being (1, N) or (, N) and similarly inputs/outputs being (1,N) or (, N) before choosing (1,N)
Then once it started running without errors grads were not correct.
Weights were nan. Not good.
Thus I made the test case that I passed through! 
Found a bunch of random errors in misindexing (using i twice :sob:, shape misconceptions, etc)
Then realized I forgot to negate the MSE derivative and was doing gradient ascent
Things were also naning out (overflow) in other areas before realizing i might have made my initial weights too highe
and it works now! I'm so happy
'''
import numpy as np
import math
import time
import random
import math
import numpy as np
def make_layer(prev_size, next_size):
    layer = {
        "weights" :  (np.random.rand( * (prev_size, next_size)) - 0.5)/5, #[[0.1] * next_size] * prev_size,
        "biases" : (np.random.rand( *(1, next_size)) - 0.5)/5, # [0.1] * next_size,
        "inputs" : None,
        "outputs" : None,
        "w_grads": np.ones((prev_size, next_size)) / 10, #[],
        "I_grads": np.ones((1, prev_size)) / 10, #[],
        "O_grads" : np.ones((1, next_size)) / 10, #[],
    }
    return layer

# def matmul(a, b):
#     ret = [[]] * len (a) * len(b[0])
#     print(ret)
#     for r in range(len(a)):
#         for c in range(len(b[0])):
#             ret[r] = a * b + c


def relu(x):
    def relu_one(k):
        if k <= 0:
            return 1e-2 * k
        else:
            return k
    return np.vectorize(lambda x: relu_one(x))(x)

class Model2():
        
    def __init__(self, input_size, output_size, hidden_layers):
        '''
        Takes list of # of things in their layers.
        Input_size and output_size are the number of input/output neurons
        hidden layers is the sizes of the hidden layers
        '''

        self.layer_sizes = hidden_layers
        self.layers = []
        self.debug=False

        
        prev_size = input_size
        
        for hidden_layer in hidden_layers:
            self.layers.append(make_layer(prev_size, hidden_layer))
            prev_size  = hidden_layer
        self.layers.append(make_layer(prev_size, output_size))

   

    def fd(self, x):
        '''f pass with input. '''
        for i in range(len(self.layers)):
            # print("layer onput", i, x)
            # print("w", self.layers[i]['weights'])
            # print("b", self.layers[i]['biases'])
            self.layers[i]['inputs'] = x
            x = x @ self.layers[i]['weights']

            x += self.layers[i]['biases']
            if i != len(self.layers) - 1:
                relu(x)            
            self.layers[i]['outputs'] = x

        return x

    def train_epoch(self, x, y, lr=10**-2, timer=False, batch_timer = False):
        '''
        f pass and then gradient descent.
        y: the goal. In sparese tensor 
        lr: how quick it learns
        '''
        if timer:
            start_time = time.perf_counter() 
        full_start = time.perf_counter()

        losses = []

        for input_idx in range(len(y)):
            pred = self.fd(x[input_idx].reshape(1, len(x[input_idx])))

            batch_start_time = time.perf_counter()
            if timer:
                print(f"Elapsed time for fd pass: { time.perf_counter()  - start_time} seconds")
                start_time = time.perf_counter() 
           
            y = np.array(y)

            
            loss = np.sum((pred - y[input_idx]) ** 2)
            # num_correct += np.sum(np.argmax(pred, axis=1) == np.argmax(y[i], axis=1))
            losses.append(loss)
            for u in range(len(self.layers) -1, -1, -1):
                # print("layer", u)
                layer = self.layers[u]
                if u == len(self.layers) - 1:
                    # print("manual def of o")
                    # print(y[input_idx])
                    # print(layer['outputs'][0])
                    layer['O_grads'] = [np.array([-2 * (y[input_idx][j] - layer['outputs'][0][j] ) for j in range(len(y[input_idx]))])]
                else:
                    layer['O_grads'] = self.layers[u + 1]['I_grads'] #+ puts the back in backaprop 

                # print("o grad", layer['O_grads'])
                # print("inputs", layer['inputs'])
                w_grad = np.zeros_like(layer['weights'])
                for i in range(len(layer['weights'])):
                    for j in range(len(layer['weights'][0])):
                        w_grad[i][j] = layer['O_grads'][0][j] * layer['inputs'][0][i] 

                #Calc O grads for next layer
                if u != 0:
                    i_grads = np.zeros_like(layer['inputs'])
                    # for i in range(1): #only one row
                    for j in range(len(layer['inputs'][0])):
                        for m in range(len(layer['weights'][0])):
                            # print("O grads", layer['O_grads'][0][m])
                            # print("Weigths", layer['weights'][j][m])
                            # print("^updating", j)
                            i_grads[0][j] += layer['O_grads'][0][m] * layer['weights'][j][m]

                    layer['I_grads'] = i_grads
                else:
                    layer['I_grads'] = "u done goofed."

                # print("I grads", layer['I_grads'])

                layer['weights'] -= lr * w_grad

                layer['biases'] -= (lr * layer['O_grads'][0]).reshape(layer['biases'].shape)
                # print("w_grad", w_grad)
            if timer:
                print(f"Elapsed time for gradient updates { time.perf_counter()  - start_time} seconds")
                start_time = time.perf_counter() 
            if batch_timer:
                print(f"Elapsed time for one batch: { time.perf_counter()  - batch_start_time} seconds")
                start_time = time.perf_counter()
        print(f" Avg loss: {sum(losses)/len(losses)}")
        print(f"Elapsed time for one epoch: { time.perf_counter()  - full_start} seconds")

        return losses

    


real_test = []
testin = np.array([[1, 2,3]], dtype=float).reshape(1, 3)
testw1 = [[4, 3], [2, 0], [1, 2]]
testb1 = [[5,6]]
testw2 = [[2, 3], [1,1]]
testb2 = [[2,4]]
testout = np.array([[54, 65]], dtype=float)
real_testmodel = Model2(3, 2, [2])

real_testmodel.layers[0]['weights'] = np.array(testw1, dtype=float)
real_testmodel.layers[1]['weights'] = np.array(testw2, dtype=float)


real_testmodel.layers[0]['biases'] = np.array(testb1, dtype=float)
real_testmodel.layers[1]['biases'] = np.array(testb2, dtype=float)

print(real_testmodel.layers[0]['weights'])

real_testmodel.train_epoch(testin, testout)

