"""Trains the deep symbolic regression architecture on given functions to produce a simple equation that describes
the dataset."""

import pickle
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from utils import pretty_print, functions
from utils.symbolic_network import SymbolicNetL0, SymbolicNet
from utils.regularization import L12Smooth  #, l12_smooth
from utils.l2l import *
from inspect import signature
import time
import argparse
import json
import traceback
import copy

from feynman_ai_equations import wave_exp, equation_dict
from benchmark import *

N_SUPPORT, N_QUERY = 10, 10


class Benchmark(BaseBenchmark):
    """Benchmark object just holds the results directory (results_dir) to save to and the hyper-parameters. So it is
    assumed all the results in results_dir share the same hyper-parameters. This is useful for benchmarking multiple
    functions with the same hyper-parameters."""

    def __init__(self, results_dir, n_layers=2, reg_weight=5e-3, inner_learning_rate=1e-2, outer_learning_rate=1e-2,
                 n_epochs1=10001, x_dim=1, inner_steps=1, m=1, exp_number=None, ood=False, train_mode='joint', equation_dict=None):
        """try to learn multiple functions within the same experiment
        for now assume they all take the same number of input variables
        additional arg x_dim
        """
        super().__init__(results_dir, n_layers, reg_weight, None, n_epochs1, None, m) # None from learning_rate and n_epochs2
        # meta-learning stuff: shared net and
        self.x_dim = x_dim
        width = len(self.activation_funcs)
        n_double = functions.count_double(self.activation_funcs)
        self.net = SymbolicNetL0(self.n_layers, funcs=self.activation_funcs,
                                 initial_weights=[
                                  # kind of a hack for truncated normal
                                  torch.fmod(torch.normal(0, init_sd_first, size=(x_dim, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_middle, size=(width, width + n_double)), 2),
                                  torch.fmod(torch.normal(0, init_sd_last, size=(width, 1)), 2)
                               ])
        self.inner_learning_rate = inner_learning_rate
        self.outer_learning_rate = outer_learning_rate
        self.train_mode = train_mode
        assert equation_dict is not None
        self.equation_dict=equation_dict
        self.inner_steps = inner_steps

    def meta_learn(self, func_names, trials, val_func_names=None):
        """Meta-train the EQL network on data generated by the given functions.
        Arguments:
            func_names: list of strings that describes the functions
            trials: number of trials to train from scratch. Will save the results for each trial.
        """
        opt = optim.Adam(self.net.parameters(), self.outer_learning_rate)

        equations = dict()
        train_losses = dict()
        val_eq = dict()
        val_losses = dict()
        for func_name in func_names:
            equations[func_name] = []
            train_losses[func_name] = []
        if val_func_names is not None:
            for val_func_name in val_func_names:
                val_eq[val_func_name] = []
                val_losses[val_func_name] = []

        if self.train_mode == "maml":
            # ------------- each iteration is one MAML outer loop
            for counter in range(self.n_epochs1):
                verbose = (counter + 1) % 250 == 0
                opt.zero_grad()
                eval_loss = 0
                for func_name in func_names:
                    func = self.equation_dict[func_name]
                    assert self.x_dim == len(signature(func).parameters)
                    # adapt to func
                    eql_for_func = self.adapt(func, func_name, verbose, equations, train_losses)
                    # eval task performance
                    x, y = generate_data(func, N_QUERY)
                    inputs, labels = x, y
                    eval_loss += self.get_loss(eql_for_func, inputs, labels)
                eval_loss.backward()
                # Average the accumulated gradients and optimize
                for p in self.net.parameters():
                    p.grad.data.mul_(1.0 / len(func_names))
                opt.step()
                if val_func_names is not None:
                    # Validation step
                    for val_func_name in val_func_names:
                        func = self.equation_dict[val_func_name]
                        eql_for_func = self.adapt(func, val_func_name, verbose)
                        x, y = generate_data(func, N_QUERY)
                        inputs, labels = x, y
                        eval_loss += self.get_loss(eql_for_func, inputs, labels)
                        val_losses[val_func_name].append(eval_loss.item())
        if self.train_mode == "joint":
            # -------------------- joint training
            for counter in range(self.n_epochs1):
                verbose = (counter + 1) % 250 == 0
                for func_name in func_names:
                    # get function, do fwd pass, compute loss
                    func = self.equation_dict[func_name]
                    inputs, labels = generate_data(func, N_SUPPORT + N_QUERY)
                    loss = self.get_loss(self.net, inputs, labels)
                    # bwd pass
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if verbose:
                        with torch.no_grad():
                            weights = self.net.get_weights()
                            expr = pretty_print.network(weights, self.activation_funcs, var_names[:self.x_dim])
                            print(expr)
                            equations[func_name].append(expr)
                            train_losses[func_name].append(loss)
                # validate
                if val_func_names is not None: # Validation step
                    for val_func_name in val_func_names:
                        # deep copy self.net so that we don't see val functions during training
                        model = copy.deepcopy(self.net)
                        func = self.equation_dict[val_func_name]
                        inputs, labels = generate_data(func, N_SUPPORT)
                        # adapt
                        loss = self.get_loss(model, inputs, labels)
                        # bwd pass
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        # eval
                        inputs, labels = generate_data(func, N_QUERY)
                        eql_val = self.get_loss(model, inputs, labels)
                        val_losses[val_func_name].append(eql_val.item())
        for func_name in func_names:
            # ----------------------------- write results to disk
            fi = open(os.path.join(self.results_dir, 'eq_summary_{}.txt'.format(func_name)), 'w')
            fi.write("\n{}\n".format(func_name))
            for expr in equations[func_name]:
                fi.write("%s\n" % (str(expr)))
            fi.close()
            np.save(os.path.join(self.results_dir, 'train_curve_{}'.format(func_name)), train_losses[func_name])
        for val_func_name in val_func_names:
            np.save(os.path.join(self.results_dir, 'val_curve_{}'.format(val_func_name)), val_losses[val_func_name])

    def get_loss(self, model, inputs, labels):
        # MSE loss
        criterion = nn.MSELoss()
        outputs = model(inputs)
        mse_loss = criterion(outputs, labels)
        # norm 1/2 reg
        regularization = L12Smooth()
        reg_loss = regularization(model.get_weights_tensor())
        loss = mse_loss + self.reg_weight * reg_loss
        return loss

    def adapt(self, func, func_name='', verbose=False, equations=None, train_losses=None):
        if verbose:
            print("****adapting to function {}****".format(func_name))

        # these should probably be command-line argument
        first_order = False; allow_unused = False; allow_nograd = False; second_order = True

        x, y = generate_data(func, N_SUPPORT)
        inputs, labels = x, y

        # clone module and specify adaptation params
        learner = clone_module(self.net)
        diff_params = [p for p in learner.parameters() if p.requires_grad]

        # ---------------------------------begin learn2learn excerpt to compute gradients
        for _ in range(0, self.inner_steps):
            loss = self.get_loss(learner, inputs, labels)
            if allow_nograd:
                # Compute relevant gradients
                diff_params = [p for p in learner.parameters() if p.requires_grad]
                grad_params = grad(loss,
                                   diff_params,
                                   retain_graph=second_order,
                                   create_graph=second_order,
                                   allow_unused=allow_unused)
                gradients = []
                grad_counter = 0

                # Handles gradients for non-differentiable parameters
                for param in learner.parameters():
                    if param.requires_grad:
                        gradient = grad_params[grad_counter]
                        grad_counter += 1
                    else:
                        gradient = None
                    gradients.append(gradient)
            else:
                try:
                    gradients = grad(loss,
                                     learner.parameters(),
                                     retain_graph=second_order,
                                     create_graph=second_order,
                                     allow_unused=allow_unused)
                except RuntimeError:
                    traceback.print_exc()
                    print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

            # Update the module
            learner = self.maml_update(learner, self.inner_learning_rate, gradients)
        adapted_learner = learner
        # -------------------------------------------------------------------------------end learn2learn excerpt
        if verbose:
            with torch.no_grad():
                weights = learner.get_weights()
                expr = pretty_print.network(weights, self.activation_funcs, var_names[:self.x_dim])
                print(expr)
                if equations is not None:
                    equations[func_name].append(expr)
                if train_losses is not None:
                    train_losses[func_name].append(loss)

        return adapted_learner

    def maml_update(self, model, lr, grads=None):
        """
        [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py)
        **Description**
        Performs a MAML update on model using grads and lr.
        The function re-routes the Python object, thus avoiding in-place
        operations.
        NOTE: The model itself is updated in-place (no deepcopy), but the
              parameters' tensors are not.
        **Arguments**
        * **model** (Module) - The model to update.
        * **lr** (float) - The learning rate used to update the model.
        * **grads** (list, *optional*, default=None) - A list of gradients for each parameter
            of the model. If None, will use the gradients in .grad attributes.
        **Example**
        ~~~python
        maml = l2l.algorithms.MAML(Model(), lr=0.1)
        model = maml.clone() # The next two lines essentially implement model.adapt(loss)
        grads = autograd.grad(loss, model.parameters(), create_graph=True)
        maml_update(model, lr=0.1, grads)
        ~~~
        """
        if grads is not None:
            params = list(model.parameters())
            if not len(grads) == len(list(params)):
                msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
                msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
                print(msg)
            for p, g in zip(params, grads):
                if g is not None:
                    p.update = - lr * g
        return update_module(model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the EQL network.")
    parser.add_argument("--train_mode", type=str, help="train mode: joint or maml?", required=True)
    parser.add_argument("--results-dir", type=str, default='results/benchmark/test')
    parser.add_argument("--n-layers", type=int, default=2, help="Number of hidden layers, L")
    parser.add_argument("--reg-weight", type=float, default=5e-3, help='Regularization weight, lambda')
    parser.add_argument('--inner_learning_rate', type=float, default=1e-2, help='inner learning rate for training')
    parser.add_argument('--outer_learning_rate', type=float, default=1e-3, help='outer learning rate for training')
    parser.add_argument('--inner_steps', type=int, default=1)
    parser.add_argument("--n-epochs1", type=int, default=20001, help="Number of epochs to train the first stage")
    parser.add_argument("--m", type=int, default=1, help="Increase Number of Activation Functions")
    parser.add_argument("--exp_number", type=int, default=1, help="Which Combination of Tasks to Use")
    parser.add_argument("--ood", action="store_true")

    args = parser.parse_args()
    kwargs = vars(args)
    assert kwargs['train_mode'] in {'joint', 'maml'}
    print(kwargs)

    results_dir = kwargs['results_dir']
    results_dir = os.path.join(results_dir, "train_{}_m_{}_exp_{}".format(kwargs['train_mode'], kwargs['m'], kwargs['exp_number']))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    kwargs['results_dir'] = results_dir
    meta = open(os.path.join(results_dir, 'args.txt'), 'a')
    meta.write(json.dumps(kwargs))
    meta.close()

    val_func_names = None
    if kwargs['exp_number'] == 1:
        func_names = ["gaussian1"]
    elif kwargs['exp_number'] == 3:
        func_names = ["id", "gaussian1",  "exp"]
    elif kwargs['exp_number'] == 5:
        func_names = ["id", "gaussian1",  "exp", "sin", "f1"]
    elif kwargs['exp_number'] == 7:
        func_names = ["id", "gaussian1",  "exp", "sin", "f1", "f2"]
    elif kwargs['exp_number'] == 9:
        func_names = ['wave1', 'wave2', 'wave3', 'wave4', 'wave5', 'wave6']
        val_func_names = ['wave7']
    elif kwargs['exp_number'] == 11:
        func_names = ['motion1', 'motion2', 'motion3', 'motion4', 'motion5', 'motion6']
        val_func_names = ['motion7']
    else:
        func_names = None

    if kwargs['exp_number'] in {1, 3, 5, 7, 9, 11}:
        kwargs['equation_dict'] = equation_dict
    if kwargs['exp_number'] == 13:
        number_train=10
        number_val=5
        ood=kwargs['ood']
        func_names, val_func_names, equation_dict = wave_exp(number_train=number_train, number_val=number_val, ood=ood)
        kwargs['results_dir'] += "_{}_ntrain_{}_nval_ood_{}_inner_steps_{}_innerlr_{}_outerlr_{}".format(number_train,number_val,ood,kwargs['inner_steps'],kwargs['inner_learning_rate'],kwargs['outer_learning_rate'])
        kwargs['equation_dict'] = equation_dict

    bench = Benchmark(**kwargs)

    bench.meta_learn(func_names=func_names, trials=10, val_func_names=val_func_names)
