import numpy as np
import math
import time

import math
import numpy as np
class Operation(object):
    """ All `Operations` represent two-variable mathematical functions, e.g. +,-,*,/,** 
        __call__ accepts two `Number` objects, and returns a Python-numeric result (int, float)
    """
    def partial_a(self):
        """ Computes the partial derivative of this operation with respect to a: d(op)/da"""
        raise NotImplementedError

    def partial_b(self):
        """ Computes the partial derivative of this operation with respect to b: d(op)/db"""
        raise NotImplementedError

    def __call__(self, a, b):
        """ Computes the forward pass of this operation: op(a, b) -> output""" 
        raise NotImplementedError

    def backprop(self, grad, recursion=True, should_print=False):
        """ Calls .backprop for self.a and self.b, passing to it dF/da and
            dF/db, respectively, where F represents the terminal `Number`-instance node 
            in the computational graph, which originally invoked `F.backprop().
             
            (In short: F is the variable with respect to which all of the partial derivatives 
            are being computed, for this back propagation)
            
            Parameters
            ----------
            grad : Union[int, float]
                dF/d(op) - The partial derivative of F with respect to the output of this operation.

            recursion: Whether or not we should keep computing afterwards. 
            """
        self.a.backprop(self.partial_a() * grad, recursion, should_print)  # backprop: dF/d(op)*d(op)/da -> dF/da
        self.b.backprop(self.partial_b() * grad, recursion, should_print)
        
    def backprop_single(self, grad):
        self.a.backprop(self.partial_a() * grad, False)  # backprop: dF/d(op)*d(op)/da -> dF/da
        self.b.backprop(self.partial_b() * grad, False)        
        
    def null_gradients(self):
        for attr in self.__dict__:
            var = getattr(self, attr)
            if hasattr(var, 'null_gradients'):
                var.null_gradients()


class Add(Operation):
    def __repr__(self): return "+"

    def __call__(self, a, b):
        """ Adds two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return a.data + b.data
    
    def partial_a(self):
        """ Returns d(a + b)/da """
        return 1
    
    def partial_b(self):
        """ Returns d(a + b)/db """
        return 1


class Multiply(Operation):
    def __repr__(self): return "*"

    def __call__(self, a, b):
        """ Nultiplies two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return a.data * b.data
    
    def partial_a(self):
        """ Returns d(a * b)/da as int or float"""
        return self.b.data
 
    def partial_b(self):
        """ Returns d(a * b)/db as int or float"""
        return self.a.data
 
class Subtract(Operation):
    def __repr__(self): return "-"

    def __call__(self, a, b):
        """ Subtracts two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return a.data - b.data
    
    def partial_a(self):
        """ Returns d(a - b)/da as int or float"""
        return 1
    
    def partial_b(self):
        """ Returns d(a - b)/db as int or float"""
        return -1


class Divide(Operation):
    def __repr__(self): return "/"

    def __call__(self, a, b):
        """ Divides two `Number` instances.
            
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return a.data / b.data
    
    def partial_a(self):
        """ Returns d(a / b)/da as int or float"""
        return 1/self.b.data
    
    def partial_b(self):
        """ Returns d(a / b)/db as int or float"""
        # STUDENT CODE HERE
        return self.a.data * - (self.b.data**-2)


class Power(Operation):
    def __repr__(self): return "**"

    def __call__(self, a, b):
        """ Exponentiates two `Number` instances.
            Parameters
            ----------
            a : Number
            b : Number
            
            Returns
            -------
            Union[int, float] """
        self.a = a
        self.b = b
        return a.data ** b.data
    
    def partial_a(self):
        """ Returns d(a ** b)/da as int or float"""
        return self.b.data * self.a.data**(self.b.data-1)

    def partial_b(self):
        """ Returns d(a ** b)/db as int or float
            Reference: http://tutorial.math.lamar.edu/Classes/CalcI/DiffExpLogFcns.aspx
        """
        if self.a.data <= 0:
            return 0
        
        return math.log(self.a.data) * self.a.data ** self.b.data


class Number(object):
    def __repr__(self):
        return "Number({})".format(self.data)

    def __init__(self, obj, *, creator=None):
        """ Parameters
            ----------
            obj : Union[int, float, Number, NumPy-numeric]
                The numerical object used as the value of this Number
            
            creator : Optional[Operation]
                The Operation-instance that produced this Number. By specifying a `creator`,
                you are effectively setting the edge from an Operation node to this Number node,
                in the computational graph being created. This allows the back-propagation process
                to 'retrace' back through the graph.
                
                Note: creator must be specified as a named variable: i.e. Number(2, creator=ref)"""
        if not ((isinstance(obj, (Number, int, float, np.generic))) or (isinstance(obj, (np.ndarray)) and obj.ndim == 0)):
            print(obj, type(obj))
        assert (isinstance(obj, (Number, int, float, np.generic)) or (isinstance(obj, (np.ndarray)) and obj.ndim == 0))
        self.data = obj.data if isinstance(obj, Number) else obj
        self._creator = creator
        self.grad = None

    @property
    def creator(self):
        """ Number.creator is a read-only property """
        return self._creator
    
    @staticmethod
    def _op(Op, a, b):
        """_op "wraps" (i.e. mediates) all of the operations performed between `Number` instances.
           
           Parameters
           ----------
           Op : subclass of Operation class. E.g. Add or Multiply
                
           a : Union[int, float, Number]
            
           b : Union[int, float, Number]
           
           Returns
           -------
           Number
               The number produced by the creator f(a, b), where f = Op().
            """
        """ Make `a` and `b` instances of `Number` if they aren't already. """
        if not isinstance( a, Number):
            a = Number(a)
        if not isinstance(b, Number):
            b = Number(b)
        
        """ Initialize Op, using `f` as its reference"""
        f = Op()
        #Basically op is a class, and we make an istance of that class of f.
        
        """ Get the output of the operation's forward pass, which is an int or float.
            Make it ans instance of `Number`, whose creator is f. Return this result."""
        ans = Number(f(a,b), creator=f)
        return ans
    
    #All the math functions. add is when it is on the lhs, radd is when the number is on the rhs
    def __add__(self, other):
        return self._op(Add, self, other)

    def __radd__(self, other):
        return self._op(Add, other, self)

    def __mul__(self, other):
        return self._op(Multiply, self, other)

    def __rmul__(self, other):
        return self._op(Multiply, other, self)

    def __truediv__(self, other):
        return self._op(Divide, self, other)

    def __rtruediv__(self, other):
        return self._op(Divide, other, self)

    def __sub__(self, other):
        return self._op(Subtract, self, other)

    def __rsub__(self, other):
        return self._op(Subtract, other, self)

    def __pow__(self, other):
        return self._op(Power, self, other)

    def __rpow__(self, other):
        return self._op(Power, other, self)

    def __neg__(self):
        return -1*self
    
    #Comparators
    # def __eq__(self, value):
    #     if isinstance(value, Number):
    #         value = value.data
    #     return self.data == value
    
    def __ge__(self, value):
        if isinstance(value, Number):
            value = value.data
        return self.data >= value
        
    def __le__(self, value):
        if isinstance(value, Number):
            value = value.data
        return self.data <= value
        
    def __gt__(self, value):
        if isinstance(value, Number):
            value = value.data
        return self.data > value

    def __lt__(self, value):
        if isinstance(value, Number):
            value = value.data
        return self.data < value
        
    def backprop(self,grad=1, recursion=True, should_print=False):
        if should_print:
            print(f"backpropping {self.data}")
        if self.grad == None:
            self.grad = grad
        else:
            self.grad += grad
        if self.creator != None and recursion:
            self.creator.backprop(grad, should_print=should_print) #chain rule orz.


    def backprop_single(self):
        """tells its creator to backprop the two numbers that went into creating it"""
        if self.grad == None:
            self.grad = 1
        if self.creator != None:
            self.creator.backprop_single(self.grad)
                                  
    def null_gradients(self, recursive = True):
        
        self.grad = None
        if self._creator is not None and recursive:
            self._creator.null_gradients()

    #Because of COURSE dicts need hashing and the topo sort will probably be cleanest with dict
    def __eq__(self, other):
        return self is other
    
    def __hash__(self):
        return id(self)

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


def cross_entropy(x, y):
    return -np.sum(y * np.log(x)) / len(y)

def softmax(x):
    # subtract max for numerical stability
    e_x = np.vectorize(lambda x: math.e**x)(x)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

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
        
    def train_epoch(self, x, y, lr=10**-2, timer=False, batch_timer = False):
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
            loss = np.sum( (pred - y[i]) ** 2) 
            # loss = cross_entropy(pred, y[i])
            num_correct += np.sum(np.argmax(pred, axis=1) == np.argmax(y[i], axis=1))
            losses.append(loss.data)

            if timer:
                print(f"Elapsed time for mse stuff: { time.perf_counter()  - start_time} seconds")
                start_time = time.perf_counter() 
            # print("mse", mse)
            mse_sorted = topo_sort(loss)
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
            #     print(f"Layer {i} avg grad magnitude:", np.mean([abs(w.grad) for w in layer.flat]))

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