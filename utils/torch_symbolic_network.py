"""Contains the symbolic regression neural network architecture."""
from inspect import signature

import torch
import torch.nn as nn
import torch_functions as functions
import pretty_print


class SymbolicLayer(nn.Module):
    """Neural network layer for symbolic regression where activation functions correspond to primitive functions.
    Can take multi-input activation functions (like multiplication)"""
    def __init__(self, funcs=None, initial_weight=None, variable=False, init_stddev=0.1, in_dim=None):
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
        self.funcs = [func.torch for func in funcs]     # Convert functions to list of PyTorch functions
        self.n_double = functions.count_double(funcs)   # Number of activation functions that take 2 inputs
        self.n_single = self.n_funcs - self.n_double    # Number of activation functions that take 1 input

        self.out_dim = self.n_funcs + self.n_double

        if initial_weight is not None:     # use the given initial weight
            # with tf.name_scope("symbolic_layer"):
            #     if not variable:
            #         self.W = tf.Variable(self.initial_weight)
            #     else:
            #         self.W = self.initial_weight

            self.W = nn.Parameter(self.initial_weight.clone().detach())  # copies
            self.built = True
        else:
            self.W = torch.normal(mean=0.0, std=init_stddev, size=(in_dim, self.out_dim))

    def forward(self, x):  # used to be __call__
        """Multiply by weight matrix and apply activation units"""
        g = torch.matmul(x, self.W)         # shape = (?, self.size)
        output = []  # TODO: will this even work? and could probably be more efficient

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


class SymbolicLayerBias(SymbolicLayer):
    """SymbolicLayer with a bias term"""
    def __init__(self, funcs=None, initial_weight=None, variable=False, init_stddev=0.1):
        super().__init__(funcs, initial_weight, variable, init_stddev)
        self.b = None

    def build(self, in_dim):
        super().build(in_dim)

    def forward(self, x):
        """Multiply by weight matrix and apply activation units"""
        super().__call__(x)
        self.output += self.b
        return self.output


class SymbolicNet(nn.Module):
    """Symbolic regression network with multiple layers. Produces one output."""
    def __init__(self, symbolic_depth, funcs=None, initial_weights=None, initial_bias=None,
                 variable=False, init_stddev=0.1):
        super(SymbolicNet, self).__init__()

        self.depth = symbolic_depth     # Number of hidden layers
        self.funcs = funcs
        layer_in_dim = [1] + self.depth*[len(funcs)]

        if initial_weights is not None:
            layers = [SymbolicLayer(funcs=funcs, initial_weight=initial_weights[i], variable=variable,
                                    in_dim=layer_in_dim[i])
                      for i in range(self.depth)]
            self.symbolic_layers = nn.Sequential(*layers)

            self.output_weight = nn.Parameter(initial_weights[-1].clone().detach())

            # if not variable:
            #     self.output_weight = tf.Variable(initial_weights[-1])
            # else:
            #     self.output_weight = initial_weights[-1]
        else:
            # Each layer initializes its own weights
            if isinstance(init_stddev, list):
                layers = [SymbolicLayer(funcs=funcs, init_stddev=init_stddev[i], in_dim=layer_in_dim[i]) for i in range(self.depth)]
                self.symbolic_layers = nn.Sequential(*layers)
            else:
                layers = [SymbolicLayer(funcs=funcs, init_stddev=init_stddev, in_dim=layer_in_dim[i]) for i in range(self.depth)]
                self.symbolic_layers = nn.Sequential(*layers)

            # Initialize weights for last layer (without activation functions)
            # TODO: does indexing into Sequential container work?
            self.output_weight = nn.Parameter(torch.rand((self.symbolic_layers[-1].n_funcs, 1)))

    def forward(self, input):
        # Building hidden layers
        h = self.symbolic_layers(input)

        # Final output (no activation units) of network
        return torch.matmul(h, self.output_weight)

    def get_weights(self):
        """Return list of weight matrices"""
        # First part is iterating over hidden weights. Then append the output weight.
        return [self.symbolic_layers[i].get_weight() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]


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


# TODO: SymbolicCell
