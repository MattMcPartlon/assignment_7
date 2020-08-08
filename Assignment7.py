from abc import abstractmethod, ABC
import numpy as np


def Func(ABC):
    def __init__(self, lb=-1e10, ub=1e10):
        self.lb = lb
        self.ub = ub
        self.num_evals = 0

    @abstractmethod
    def _eval_func(self, x):
        pass

    def get_num_evals(self):
        return self.num_evals

    def eval_func(self, x):
        assert self.lb < x < self.ub
        self.num_evals += 1
        return self._eval_func(x)

    def eval_deriv(self, x, approx=True):
        if approx:
            self.approx_deriv(x)
        else:
            self._eval_deriv(x)

    @abstractmethod
    def _eval_deriv(self, x):
        pass

    @abstractmethod
    def approx_deriv(self, x, eps=1e-7):
        pass

    def get_lb(self):
        return self.lb

    def get_ub(self):
        return self.ub

def UnivariateFunc(Func):
    
    def __init__(self, lb = -1e10, ub = 1e10):
        super().__init__(lb,ub)
    
    @abstractmethod
    def _eval_func(self, x):
        pass
    
    def eval_deriv(self, x, approx = True):
        if approx : 
            self.approx_deriv(x)
        else:
            self._eval_deriv(x)
    
    @abstractmethod
    def _eval_deriv(self, x):
        pass

    def approx_deriv(self, x, eps = 1e-7):
        f0 = self.eval_func(x)
        f1 = self.eval_func(x+eps)
        return (f1-f0)/eps
    
    def get_lb(self):
        return self.lb
    
    def get_ub(self):
        return self.ub


def MultivariateFunc(ABC):
    def __init__(self, lb=-1e10, ub=1e10):
        super().__init__(lb,ub)

    @abstractmethod
    def _eval_func(self, x):
        pass

    def eval_deriv(self, x, approx=True):
        if approx:
            self.approx_deriv(x)
        else:
            self._eval_deriv(x)

    @abstractmethod
    def _eval_deriv(self, x):
        pass

    @abstractmethod
    def approx_deriv(self, x, eps=1e-7):
        pass


class RootFinder(ABC):
    
    def __init__(self, warn = True):
        self.warn = warn
    
    def setup_for_root_find(self, func : UnivariateFunc, x, max_evals=1e4, tol = 1e-6):
        """
        max_evals : maximum number of f evaluations to use during root finding
        tol : tolerance allowed for finding roots (x is a root if abs(f(x))<tol
        """
        self.func = func
        self.tol = tol
        self.max_evals = max_evals
        self.best_x = x
        self.best_f = self.func.eval_func(x)
    
    @abstractmethod
    def find_root(self, func : UnivariateFunc, x0 = None, *args, **kwargs):
        pass
    
    def eval_func(self, x):
        #will pad three extra alotted evaluation of f so that
        #calls to eval_func and eval_deriv in succession do not 
        #throw assertion errors
        assert not self.finishedQ() or self.func.get_num_evals() < self.max_evals
        f_x = self.func.eval_func(x)
        if self.best_f > abs(f_x):
            self.best_x = x
            self.best_f = abs(f_x)
        return f_x
    
    def eval_deriv(self, x, approx = True):
        #assert not self.finishedQ()
        return self.func.eval_deriv(x, approx = approx)
    
    def finishedQ(self):
        t1 =  self.func.get_num_evals()>self.func.get_num_evals()
        t2 = self.best_f < self.tol
        return t1 or t2
    
    def get_sol(self):
        assert self.finishedQ()
        if self.best_f>self.tol and self.warn:
            s = 'WARNING, maximum number of f_evals exceeded, '
            s+= 'but we have failed to converge on a root'
            print(s)
            print('Best solution : ',self.func.eval_func(self.best_x))
        return self.best_x
                              

class MCMC(RootFinder): #(3 points)
    # NOTE : all function evaluations must be done via super().eval_func(x)

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def find_root(self, func, x0 =None, max_evals = 1e4, tol = 1e-5, *args, **kwargs):
        """
        attempts to find the root of the function func beginning from
        point x0 
       
        @return a point x satisfying | func(x) | <= tol
        if such a point exists, else returns best solution seen during search
        (obtained by calling super().get_sol())
        """
        if x0 is None:
            x0 = self.sample_starting_point(func.get_lb(), func.get_ub())
        self.setup_for_root_find(func, x0, max_evals = max_evals, tol = tol)
        while not self.finishedQ():
            x0 = self.MCMC_step(x0, *args, **kwargs)

    #you may add additional arguments here as you wish
    def MCMC_step(self, x0):
        #implement one MCMC step to change the current guess x0
        #you may want to add additional parameters such as,
        # an acceptance probability for |f| increasing moves
        # you may even make this acceptance probability depend on the 
        # number of f evaluations, or the current iteration (which would
        #make this simulated annealing)
        #you may wish to try larger changes for x0 when |f(x0)| is far from 0,
        #and smaller changes when it is close... the step size may be random
        pass
    
    def sample_starting_point(self, lb, ub, n_points=100):
        #OPTIONAL - before beginning root-finding,
        #you may wish to sample a starting point x0 from a 
        #distribution over f(x) with x in range lb to ub
        # to sample over a distribution favoring mass on the 
        #roots of a function, we create a distribution D where
        # D(x) ~ exp(-|f(x)|/r)/Z, where Z is a normalizing constant,
        # and r is a scaling constant to prevent overflow.
        # note that points x where f(x) ~ 0 have the largest mass
        
        #REMOVE IF YOU WISH TO IMPLEMENT SAMPLING for starting point
        return (ub+lb)/2


class BracketNBisect(RootFinder): #(4 points)
    # NOTE : all function evaluations must be done via super().eval_func(x)

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def find_root(self, func, lb, ub, max_evals = 1e4, tol = 1e-5 ):
        """
        attempts to find the root of the function func
        in the interval [lb,ub]
        @param lb lower bound on x for func(x) evaluations
        @param ub upper bound on x for func(x) evaluations
        @return a point x such that lb<=x<=ub satisfying | func(x) | <= tol
        if such a point exists, else returns best solution seen during search
        (obtained by calling super().get_sol())
        """
        self.setup_for_root_find(func, lb+ub/2, max_evals = max_evals, tol = tol)
        bracket = None
        while bracket is None and not self.finishedQ():
            #attempt to bracket a root of func
            pass
        
        #if bracket is still None at this point, then 
        #we could not bracket a root in the alotted num f evaluations
        #we will return the best value seen during bracket search
        
        if bracket is None:
            return self.get_sol()
        
        start, end = bracket
        while not self.finishedQ():
            #attempt to find a root of f via the bisection method in the interval
            # start, end via bisection
            pass
        
        return self.get_sol()
    
    def bracket_root(self, lb, ub, n_bracks):
        """
        attempts to find an interval [start,end] such that
        sign(func(start)) == -sign(f(end))
        @param lb lower bound on x for func(x) evaluations
        @param ub upper bound on x for func(x) evaluations
        @return a pair of points (start, end) such that
        sign(f(start))!=sign(f(end)), or returns None if no such
        points can be found in the alotted time.
        """
        intervals = self.get_intervals(lb, ub, n_bracks)
        curr_lb, curr_ub = intervals[0]
        sign_change = self.eval_func(curr_lb)*self.eval_func(curr_ub)<0
        while not self.finishedQ() and not sign_change:
            #TODO
            pass
        
    
    def get_intervals(self, lb, ub, n):
        """
        return n evenly spaced intervals between lb and ub
        """
        intervals = np.empty((n,2))
        cutoffs = np.linspace(lb, ub, n+1)
        intervals[:,0] = cutoffs[:-1]
        intervals[:,1] = cutoffs[1:]
        return intervals
    
    def bisect(self, lb, ub):
        """
        attempt to find a root in the range [lb,ub], via the bisection method
        #NOTE : does not need a return, as the solution is stored by the 
        super class (assuming eval_func is called via super().eval_func )
        """

        while not self.finishedQ():
            #attempt to find a root via the bisection method
            pass


class NewtonRaphson(RootFinder): #4 points
    """
    Find the root of a function via the newton - raphson method of truncating the
    taylor series
    """

    # NOTE : all function evaluations must be done via super().eval_func(x)
    #pass any parameters you need to init
    def __init__(self):
        super.__init__()
        
    def find_root(self, func, x0, max_evals = 1e4, tol = 1e-5 ):
        """
        attempts to find the root of the function func via the Newton-Raphson method
        @return a point x such that | func(x) | <= tol
        if such a point exists, else returns best solution seen during search 
        (obtained by calling super().get_sol())
        """
        self.setup_for_root_find(func, (func.lb+func.ub)/2, max_evals = max_evals, tol = tol)
        while not self.finishedQ():
            x0 = self.newton_raphson_step(x0)
        return super.get_sol()
    
    def newton_raphson_step(self, x0):
        pass
        #TODO - implement one step of Newton - Raphson method




"""
Represents the univariate function f(x + delta*d) w.r.t the variable delta,
when x and d are given. 

We can think of optimizing a multivariate function as follows :
(1) given f, x \in R^n, find the direction d \in R^n of greatest decrease for f
this direction is equal to -1 * gradient(f(x)).
(2) find delta minimizing f(x+ delta *d)

note that  f(x+ delta *d) is a univariate function! therefore, we just have to
minimize a function of a single variable when deciding how to change f.

Finding such a delta during a step of gradient descent is called
"performing a line search" you will implement two line search classes,
described below

To be used in gradient descent with multivariate function as input
"""

class DirectionalFunc(UnivariateFunc):
    
    def __init__(self, func, x, lb = -1e10, ub = 1e10):
        super().__init__(lb,ub)
        self.func = func
        self.x = x
        self.direction = self.func.eval_deriv(x)
        
    def _eval_func(self, delta):
        return self.func.eval_func(self.x+self.direction*delta)

    #should not need to eval the derivative here
    def _eval_deriv(self, x):
        pass

class LineSearch(ABC):

    def minimize(self, func, curr_x):
        dfunc = DirectionalFunc(func, curr_x)
        d = dfunc.direction
        x0 = curr_x
        if isinstance(curr_x,np.ndarray):
            x0 = np.copy(curr_x)
        #task is to find delta minimizing
        #f(x) + delta*d
        delta = self.find_delta(dfunc)
        return x0+delta*d

    @abstractmethod
    def find_delta(self, dfunc):
        #TODO
        pass

class BinaryLineSearch(LineSearch): #2 points

    def find_delta(self, dfunc, n_eval_limit = 100, tol=1e-5):
        """
        attempts to find delta via a binary search.
        the idea is as follows:

        assume at the start,  ub = 1, lb = 0.
        if f(ub)<f(lb), then lb = ub, ub = 2*ub.
        otherwise, lb = lb, ub = (lb+ub)/2.

        this continues until abs(lb-ub)<tol,
        or we have exceeded the maximum number of
        alotted evaluations.


        :param dfunc:
        :return:
        """
        #TODO
        pass


class RootFindLineSearch(LineSearch): #4 points

    def find_delta(self, dfunc, n_eval_limit=100, tol=1e-5):
        """
        attempts to find delta via finding a root of
        f(x + delta * grad(f(x)).
        :param dfunc:
        :return:
        """
        #TODO
        pass


class GradientDescent: #6 points
    """
    Implement gradient descent on a function f via passing the appropriate func,
    and a line search method to use for each gradient descent step.

    the method should accept a maximum number of evaluations,
    tolerance parameter for the gradient, and a tolerance parameter
    for the change in the function value.
    """

    def __init__(self):
        pass



"""
6 points

Implement a linear regression class which takes as input a matrix X of 
points, and a vector y of labels, and finds a vector w and bias b such that
||(Aw +b) - y||^2 is minimal. 

NOTE: you should append a column of 1's to A so that the bias b 
is found automagically.

Run this on a one dimensional data set and plot your results
"""


"""
up to 10 bonus points 

Implement a more sophisticated classifier and optimize it using the 
modules you have built up in the homework. You must apply it to a dataset
of your choosing and discuss the results.
"""





