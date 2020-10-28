"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset."""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import pretty_print, functions
from utils.symbolic_network import SymbolicNetL0
import time
import argparse
import json

from feynman_ai_equations import equation_dict
from benchmark import *


class Benchmark(BaseBenchmark):
    """Benchmark object just holds the results directory (results_dir) to save to and the hyper-parameters. So it is
    assumed all the results in results_dir share the same hyper-parameters. This is useful for benchmarking multiple
    functions with the same hyper-parameters."""
    def __init__(self, results_dir, n_layers=2, reg_weight=1e-2, learning_rate=1e-2,
                 n_epochs1=10001):
        super().__init__(results_dir, n_layers, reg_weight, learning_rate, n_epochs1,)
        self.activation_funcs = [[
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 10,
            # *[functions.Square()] * 4,
            # *[functions.Sin()] * 2,
            *[functions.Exp()] * 2,
            *[functions.Sigmoid()] * 2,
            # *[functions.Reciprocal(1.0)] * 2,
            # *[functions.Product(1.0)] * 2,
        ],
        [
            *[functions.Constant()] * 2,
            *[functions.Identity()] * 4,
            *[functions.Square()] * 4,
            # *[functions.Sin()] * 2,
            # *[functions.Exp()] * 2,
            *[functions.Sigmoid()] * 2,
            *[functions.Division(1.0)] * 2,
            # *[functions.Product(1.0)] * 2,
        ],
        # [
        #     # *[functions.Constant()] * 2,
        #     # *[functions.Identity()] * 4,
        #     # *[functions.Reciprocal(1.0)] * 2,
        #     *[functions.Division(1.0)] * 2,
        # ]
        ]

    def train(self, func, func_name='', trials=1, func_dir='results/test'):
        """Train the network to find a given function"""

        x, y = generate_data(func, N_TRAIN)
        # x_val, y_val = generate_data(func, N_VAL)
        x_test, y_test = generate_data(func, N_TEST, range_min=DOMAIN_TEST[0], range_max=DOMAIN_TEST[1])

        # Setting up the symbolic regression network
        x_dim = len(signature(func).parameters)  # Number of input arguments to the function

        if not any(isinstance(el, list) for el in self.activation_funcs):
            self.activation_funcs = [self.activation_funcs] * self.n_layers

        # width = [len(func_i) for func_i in self.activation_funcs]
        # n_double = [functions.count_double(func_i) for func_i in self.activation_funcs]
        # initial_weights=[
        #     # kind of a hack for truncated normal
        #     torch.fmod(torch.normal(0, init_sd_first, size=(x_dim, width[0] + n_double[0])), 2),
        #     torch.fmod(torch.normal(0, init_sd_middle, size=(width[0], width[1] + n_double[1])), 2),
        #     torch.fmod(torch.normal(0, init_sd_last, size=(width[1], 1)), 2)
        # ]
        initial_weights = None

        # Arrays to keep track of various quantities as a function of epoch
        loss_list = []          # Total loss (MSE + regularization)
        error_list = []         # MSE
        reg_list = []           # Regularization
        error_test_list = []    # Test error

        error_test_final = []
        eq_list = []

        for trial in range(trials):
            print("Training on function " + func_name + " Trial " + str(trial+1) + " out of " + str(trials))

            criterion = nn.MSELoss()

            loss_val = np.nan
            # Restart training if loss goes to NaN (which happens when gradients blow up)
            while np.isnan(loss_val):

                # reinitialize for each trial
                net = SymbolicNetL0(self.n_layers, in_dim=x_dim, funcs=self.activation_funcs,
                                    initial_weights=initial_weights)
                optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)

                # adapative learning rate
                lmbda = lambda epoch: 0.1 * epoch
                scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

                for param_group in optimizer.param_groups:
                    print("Learning rate: %f" % param_group['lr'])

                t0 = time.time()

                # First stage of training, preceded by 0th warmup stage
                for epoch in range(self.n_epochs1 + 2000):
                    inputs, labels = x, y

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    outputs = net(inputs)

                    mse_loss = criterion(outputs, labels)
                    reg_loss = net.get_loss()
                    loss = mse_loss + self.reg_weight * reg_loss

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.01)
                    optimizer.step()

                    if epoch % self.summary_step == 0:
                        error_val = mse_loss.item()
                        reg_val = reg_loss.item()
                        loss_val = error_val + self.reg_weight * reg_val
                        print("Epoch: %d\tTotal training loss: %f\tReg loss: %f" % (epoch, loss_val, reg_val))
                        error_list.append(error_val)
                        reg_list.append(reg_val)
                        loss_list.append(loss_val)

                        # TODO: error test val
                        # loss_val, error_val, reg_val, = sess.run((loss, error, reg_loss), feed_dict=feed_dict)
                        # error_test_val = sess.run(error_test, feed_dict={x_placeholder: x_test})
                        # print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (i, loss_val, error_test_val))

                        # error_list.append(error_val)
                        # error_test_list.append(error_test_val)
                        if np.isnan(loss_val):  # If loss goes to NaN, restart training
                            break

                    if epoch == 100:
                        scheduler.step()  # lr /= 10
                        for param_group in optimizer.param_groups:
                            print("Learning rate: %f" % param_group['lr'])

                t1 = time.time()

            tot_time = t1 - t0
            print(tot_time)

            # Print the expressions
            with torch.no_grad():
                weights = net.get_weights()
                expr = pretty_print.network(weights, self.activation_funcs, var_names[:x_dim])
                print(expr)

            # Save results
            trial_file = os.path.join(func_dir, 'trial%d.pickle' % trial)
            results = {
                "weights": weights,
                "loss_list": loss_list,
                "error_list": error_list,
                "reg_list": reg_list,
                "error_test": error_test_list,
                "expr": expr,
                "runtime": tot_time
            }
            with open(trial_file, "wb+") as f:
                pickle.dump(results, f)

            # error_test_final.append(error_test_list[-1])
            eq_list.append(expr)

        return eq_list, error_test_final


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--results-dir", type=str, default='results/benchmark/test')
    parser.add_argument("--n-layers", type=int, default=2, help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float, default=1e-2, help='Regularization weight, lambda')
    parser.add_argument('--learning-rate', type=float, default=1e-1, help='Base learning rate for training')
    parser.add_argument("--n-epochs1", type=int, default=10001, help="Number of epochs to train the first stage")

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)

    if not os.path.exists(kwargs['results_dir']):
        os.makedirs(kwargs['results_dir'])
    meta = open(os.path.join(kwargs['results_dir'], 'args.txt'), 'a')
    meta.write(json.dumps(kwargs))
    meta.close()

    bench = Benchmark(**kwargs)

    func_name = "exp1"
    # bench.benchmark(equation_dict[func_name], func_name=func_name, trials=10)
    # bench.benchmark(lambda x: 1/x, func_name="1divx", trials=10)
    bench.benchmark(lambda x, y: x / y, func_name="xdivy", trials=10)
    # bench.benchmark(lambda x: x, func_name="x", trials=10)
    # bench.benchmark(lambda x: x**2, func_name="x^2", trials=20)
    # bench.benchmark(lambda x: x**3, func_name="x^3", trials=20)
    # bench.benchmark(lambda x: np.sin(2*np.pi*x), func_name="sin(2pix)", trials=20)
    # bench.benchmark(lambda x: np.exp(x), func_name="e^x", trials=20)
    # bench.benchmark(lambda x, y: x*y, func_name="xy", trials=20)
    # bench.benchmark(lambda x, y: np.sin(2 * np.pi * x) + np.sin(4*np.pi * y),
    #                 func_name="sin(2pix)+sin(4py)", trials=20)
    # bench.benchmark(lambda x, y, z: 0.5*x*y + 0.5*z, func_name="0.5xy+0.5z", trials=20)
    # bench.benchmark(lambda x, y, z: x**2 + y - 2*z, func_name="x^2+y-2z", trials=20)
    # bench.benchmark(lambda x: np.exp(-x**2), func_name="e^-x^2", trials=20)
    # bench.benchmark(lambda x: 1 / (1 + np.exp(-10*x)), func_name="sigmoid(10x)", trials=20)
    # bench.benchmark(lambda x, y: x**2 + np.sin(2*np.pi*y), func_name="x^2+sin(2piy)", trials=20)

    # 3-layer functions
    # bench.benchmark(lambda x, y, z: (x+y*z)**3, func_name="(x+yz)^3", trials=20)


