"""Contains the symbolic regression neural network architecture."""
from inspect import signature

import torch
import torch.nn as nn
import utils.functions as functions
import utils.pretty_print as pretty_print
import numpy as np


class SymbolicLayer(nn.Module):
    """Neural network layer for symbolic regression where activation functions correspond to primitive functions.
    Can take multi-input activation functions (like multiplication)"""
    def __init__(self, funcs=None, initial_weight=None, init_stddev=0.1, in_dim=None):
        """
        funcs: List of activation functions, using utils.functions
        initial_weight: (Optional) Initial value for weight matrix
        variable: Boolean of whether initial_weight is a variable or not
        init_stddev: (Optional) if initial_weight isn't passed in, this is standard deviation of initial weight
        """
        super().__init__()

        if funcs is None:
            funcs = functions.default_func

        self.W = None       # Weight matrix
        self.built = False  # Boolean whether weights have been initialized

        self.n_funcs = len(funcs)                       # Number of activation functions (and number of layer outputs)
        self.func_objects = funcs                       # Function objects
        self.funcs = [func.torch for func in funcs]     # Convert functions to list of PyTorch functions
        self.n_double = functions.count_double(funcs)   # Number of activation functions that take 2 inputs
        self.n_single = self.n_funcs - self.n_double    # Number of activation functions that take 1 input

        self.in_dim = in_dim
        self.out_dim = self.n_funcs + self.n_double

        if initial_weight is not None:     # use the given initial weight
            self.W = nn.Parameter(initial_weight.clone().detach())  # copies
            self.built = True
        else:
            self.W = nn.Parameter(torch.normal(mean=0.0, std=init_stddev, size=(in_dim, self.out_dim)))

        self.theta = 0.01
        self.penalty = torch.zeros(1)    # Penalty for the division term

    def forward(self, x):  # used to be __call__
        """Multiply by weight matrix and apply activation units"""
        g = torch.matmul(x, self.W)         # shape = (?, self.size)
        output = []

        in_i = 0    # input index
        out_i = 0   # output index
        # Apply functions with only a single input
        while out_i < self.n_single:
            output.append(self.funcs[out_i](g[:, in_i]))
            in_i += 1
            out_i += 1
        # Apply functions that take 2 inputs and produce 1 output
        while out_i < self.n_funcs:
            output.append(self.funcs[out_i](g[:, in_i], g[:, in_i+1]))
            in_i += 2
            out_i += 1

        # self.output = torch.stack(self.output, dim=1)
        # self.output = g
        output = torch.stack(output, dim=1)

        return output

    def get_weight(self):
        return self.W.cpu().detach().numpy()

    def get_weight_tensor(self):
        return self.W.clone()


class SymbolicLayerBias(SymbolicLayer):
    """SymbolicLayer with a bias term"""
    def __init__(self, funcs=None, initial_weight=None, variable=False, init_stddev=0.1):
        super().__init__(funcs, initial_weight, variable, init_stddev)
        self.b = None

    def forward(self, x):
        """Multiply by weight matrix and apply activation units"""
        super().__call__(x)
        self.output += self.b
        return self.output


class SymbolicNet(nn.Module):
    """Symbolic regression network with multiple layers. Produces one output."""
    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, initial_bias=None, init_stddev=0.1, in_dim=1):
        super(SymbolicNet, self).__init__()

        if any(isinstance(el, list) for el in funcs):
            self.funcs = funcs
            self.depth = len(funcs)
        else:
            self.depth = symbolic_depth  # Number of hidden layers
            self.funcs = [funcs] * symbolic_depth
        layer_in_dim = [in_dim] + [len(funcs_i) for funcs_i in self.funcs]

        if initial_weights is not None:
            layers = [SymbolicLayer(funcs=self.funcs[i], initial_weight=initial_weights[i], in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())

        else:
            # Each layer initializes its own weights
            if not isinstance(init_stddev, list):
                init_stddev = [init_stddev] * self.depth
            layers = [SymbolicLayer(funcs=self.funcs[i], init_stddev=init_stddev[i], in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            # Initialize weights for last layer (without activation functions)
            self.output_weight = nn.Parameter(torch.rand((layers[-1].n_funcs, 1)))
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input):
        h = self.hidden_layers(input)     # Symbolic_layers is nn.Sequential of all the hidden layers

        # Final output (no activation units) of network
        return torch.matmul(h, self.output_weight)

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [self.hidden_layers[i].get_weight() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]

    def get_weights_tensor(self):
        """Return list of weight matrices as tensors"""
        return [self.hidden_layers[i].get_weight_tensor() for i in range(self.depth)] + \
               [self.output_weight.clone()]


class SymbolicLayerL0(SymbolicLayer):
    def __init__(self, in_dim=None, funcs=None, initial_weight=None, init_stddev=0.1,
                 bias=False, droprate_init=0.5, lamba=1.,
                 beta=2/3, gamma=-0.1, zeta=1.1, epsilon=1e-6):
        super().__init__(in_dim=in_dim, funcs=funcs, initial_weight=initial_weight, init_stddev=init_stddev)

        self.droprate_init = droprate_init if droprate_init != 0 else 0.5
        self.use_bias = bias
        self.lamba = lamba
        self.bias = None
        self.eps = None
        
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.epsilon = epsilon

        if self.use_bias:
            self.bias = nn.Parameter(0.1*torch.ones((1, self.out_dim)))
        self.qz_log_alpha = nn.Parameter(torch.normal(mean=np.log(1 - self.droprate_init) - np.log(self.droprate_init),
                                         std=1e-2, size=(in_dim, self.out_dim)))

    def quantile_concrete(self, u):
        """Quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(u) - torch.log(1.0-u) + self.qz_log_alpha) / self.beta)
        return y * (self.zeta - self.gamma) + self.gamma

    def sample_u(self, shape):
        """Uniform random numbers for concrete distribution"""
        self.eps = torch.rand(size=shape) * (1 - 2 * self.epsilon) + self.epsilon
        return self.eps

    def sample_z(self, batch_size, sample=True):
        """Use the hard concrete distribution as described in https://arxiv.org/abs/1712.01312"""
        if sample:
            eps = self.sample_u((batch_size, self.in_dim, self.out_dim))
            z = self.quantile_concrete(eps)
            return torch.clamp(z, min=0, max=1)
        else:   # Mean of the hard concrete distribution
            pi = torch.sigmoid(self.qz_log_alpha)
            return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def get_z_mean(self):
        """Mean of the hard concrete distribution"""
        pi = torch.sigmoid(self.qz_log_alpha)
        return torch.clamp(pi * (self.zeta - self.gamma) + self.gamma, min=0.0, max=1.0)

    def sample_weights(self):
        z = self.quantile_concrete(self.sample_u((self.in_dim, self.out_dim)))
        mask = torch.clamp(z, min=0.0, max=1.0)
        return mask * self.W

    def get_weight(self):
        """Deterministic value of weight based on mean of z"""
        return self.W * self.get_z_mean()

    def loss(self):
        """Regularization loss term"""
        # print(torch.sum(self.penalty))
        return torch.sum(torch.sigmoid(self.qz_log_alpha - self.beta * np.log(-self.gamma / self.zeta))) + \
               torch.sum(self.penalty)

    def forward(self, x, sample=True):
        """Multiply by weight matrix and apply activation units"""
        if sample:
            h = torch.matmul(x, self.sample_weights())
        else:
            w = self.get_weight()
            h = torch.matmul(x, w)

        if self.use_bias:
            h = h + self.bias

        # shape of h = (?, self.n_funcs)
        self.penalty = torch.zeros(1)

        output = []
        # apply a different activation unit to each column of h
        in_i = 0    # input index
        out_i = 0   # output index
        # Apply functions with only a single input
        while out_i < self.n_single:
            if isinstance(self.func_objects[out_i], functions.Reciprocal):
                self.penalty = self.penalty + torch.max(self.theta - h[:, in_i], torch.zeros_like(h[:, in_i]))
            output.append(self.funcs[out_i](h[:, in_i]))
            in_i += 1
            out_i += 1
        # Apply functions that take 2 inputs and produce 1 output
        while out_i < self.n_funcs:
            if isinstance(self.func_objects[out_i], functions.Division):
                self.penalty = self.penalty + torch.max(self.theta - h[:, in_i+1], torch.zeros_like(h[:, in_i+1]))
            output.append(self.funcs[out_i](h[:, in_i], h[:, in_i+1]))
            in_i += 2
            out_i += 1
        output = torch.stack(output, dim=1)
        return output

    def get_weight_tensor(self):
        return self.get_weight().clone()


class SymbolicNetL0(nn.Module):
    """Symbolic regression network with multiple layers. Produces one output."""
    def __init__(self, symbolic_depth=2, in_dim=1, funcs=None, initial_weights=None, init_stddev=0.1):
        """
        Arguments:
        symbolic_depth:     Number of hidden layers. This can be ignored if funcs is given as a list of list of
                            functions - one list per hidden layer.
        in_dim:             int, input dimensionality
        funcs:              Either a list of functions (so each hidden layer uses the same functions) or a list of list
                            of functions (so each hidden layer could use different functions)
        initial_weights:    initial weights - must match the tensor size
        init_stddev:        initial weight standard deviation. this is used if initial_weights is not given
        """
        super(SymbolicNetL0, self).__init__()

        # self.funcs is made into a list of list of functions, one list for each hidden layer
        if any(isinstance(el, list) for el in funcs):
            self.funcs = funcs
            self.depth = len(funcs)
        else:
            self.depth = symbolic_depth  # Number of hidden layers
            self.funcs = [funcs] * symbolic_depth

        layer_in_dim = [in_dim] + [len(funcs_i) for funcs_i in self.funcs]  # input dimensionality for each layer
        if initial_weights is not None:
            layers = [SymbolicLayerL0(funcs=self.funcs[i], initial_weight=initial_weights[i],
                                      in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())
        else:
            # Each layer initializes its own weights
            if not isinstance(init_stddev, list):
                init_stddev = [init_stddev] * self.depth
            layers = [SymbolicLayerL0(funcs=self.funcs[i], init_stddev=init_stddev[i], in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            # Initialize weights for last layer (without activation functions)
            self.output_weight = nn.Parameter(torch.rand(size=(layers[-1].n_funcs, 1)) * 2)
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input, sample=True, reuse_u=False):
        h = self.hidden_layers(input)     # Symbolic_layers is nn.Sequential of all the hidden layers
        # Final output (no activation units) of network
        return torch.matmul(h, self.output_weight)

    def get_loss(self):
        return torch.sum(torch.stack([self.hidden_layers[i].loss() for i in range(self.depth)]))

    def get_weights(self):
        return self.get_hidden_weights() + [self.get_output_weight()]

    def get_hidden_weights(self):
        return [self.hidden_layers[i].get_weight() for i in range(self.depth)]

    def get_output_weight(self):
        return self.output_weight

    def get_weights_tensor(self):
        """Return list of weight matrices as tensors"""
        return [self.hidden_layers[i].get_weight_tensor() for i in range(self.depth)] + \
               [self.output_weight.clone()]


if __name__ == '__main__':
    n_layers = 2
    activation_funcs = [
                *[functions.Constant()] * 2,
                *[functions.Identity()] * 4,
                *[functions.Square()] * 4,
                # *[functions.Sin()] * 2,
                # *[functions.Exp()] * 2,
                # *[functions.Sigmoid()] * 2,
                # *[functions.Product()] * 2
            ]
    var_names = ["x", "y", "z"]
    
    func = lambda x: x
    x_dim = len(signature(func).parameters)  # Number of input arguments to the function
    
    N = 10
    x = torch.rand((N, x_dim)) * 2 - 1
    width = len(activation_funcs)
    n_double = functions.count_double(activation_funcs)
    
    sym = SymbolicNet(n_layers, funcs=activation_funcs,
                      # initial_weights=[torch.zeros(size=(x_dim, width + n_double)),  # kind of a hack for truncated normal
                      #                  torch.zeros(size=(width, width + n_double)),
                      #                  torch.zeros(size=(width, width + n_double)),
                      #                  torch.zeros(size=(width, 1))],
                      # initial_weights=[torch.ones(size=(x_dim, width + n_double)),  # kind of a hack for truncated normal
                      #                  torch.ones(size=(width, width + n_double)),
                      #                  torch.ones(size=(width, width + n_double)),
                      #                  torch.ones(size=(width, 1))],
                      # initial_weights=[torch.fmod(torch.normal(0, 1, size=(x_dim, width + n_double)), 2),  # kind of a hack for truncated normal
                      #                  torch.fmod(torch.normal(0, 1, size=(width, width + n_double)), 2),
                      #                  torch.fmod(torch.normal(0, 1, size=(width, width + n_double)), 2),
                      #                  torch.fmod(torch.normal(0, 1, size=(width, 1)), 2)
                      #                  ]
    )
    
    with torch.no_grad():
        weights = sym.get_weights()
        expr = pretty_print.network(weights, activation_funcs, var_names[:x_dim])
        print(expr)
    
    optimizer = torch.optim.Adam(sym.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()
    y = func(x)
    for i in range(1000):
        yhat = sym(x)
        reg = torch.tensor(0.)
        for param in sym.parameters():
            reg = reg + 0.01*torch.norm(param, 0.5)
        loss = loss_func(yhat, y) + reg
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    optimizer = torch.optim.Adam(sym.parameters(), lr=0.001)
    for i in range(1000):
        yhat = sym(x)
        reg = torch.tensor(0.)
        for param in sym.parameters():
            reg = reg + 0.01*torch.norm(param, 0.5)
        loss = loss_func(yhat, y) + reg
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        weights = sym.get_weights()
        expr = pretty_print.network(weights, activation_funcs, var_names[:x_dim])
        print(expr)

