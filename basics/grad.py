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

    def backprop(self, grad, recursion=True):
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
        self.a.backprop(self.partial_a() * grad, recursion)  # backprop: dF/d(op)*d(op)/da -> dF/da
        self.b.backprop(self.partial_b() * grad, recursion)
        
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
        # self.a.backprop()
        # self.b.backprop()
        # print(self.b.data , self.a.grad, self.a.data ,self.b.grad)
        # return self.b.data * self.a.grad + self.a.data * self.b.grad
    
    def partial_b(self):
        """ Returns d(a * b)/db as int or float"""
        return self.a.data
        # self.a.backprop()
        # self.b.backprop()
        # return self.b.data * self.a.grad + self.a.data * self.b.grad


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
        # STUDENT CODE HERE
        return self.b.data * self.a.data**(self.b.data-1)

    def partial_b(self):
        """ Returns d(a ** b)/db as int or float
            Reference: http://tutorial.math.lamar.edu/Classes/CalcI/DiffExpLogFcns.aspx
        """
        if self.a.data <= 0:
            # raise ValueError(f"Cannot compute log({self.a.data})")
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
        # if not isinstance(obj, (Number, int, float, np.generic)) or (isinstance(obj, (np.ndarray)) and obj.ndim == 0):
        #     print(obj, type(obj))
        assert isinstance(obj, (Number, int, float, np.generic)) or (isinstance(obj, (np.ndarray)) and obj.ndim == 0)
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
        # Delete this raise-error statement once you have completed your implementation of `_op`

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
    
    def __eq__(self, value):
        if isinstance(value, Number):
            value = value.data
        return self.data == value
    
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
        
    def backprop(self,grad=1, recursion=True):
    
        if self.grad == None:
            self.grad = grad
        else:
            self.grad += grad
        if self.creator != None and recursion:
            self.creator.backprop(grad) #chain rule orz.


    def backprop_single(self):
        """tells its creator to backprop the two numbers that went into creating it"""
        if self.grad == None:
            self.grad = 1
            # print("this should only print once")
        if self.creator != None:
            self.creator.backprop_single(self.grad)
                                  
    def null_gradients(self, recursive = True):
        if self.grad == None:
            return
        self.grad = None
        if self._creator is not None and recursive:
            self._creator.null_gradients()

    #Because of COURSE dicts need hashing and the topo sort will probably be cleanest with dict
    def __eq__(self, other):
        return self is other
    
    def __hash__(self):
        return id(self)
