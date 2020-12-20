# 6.883-Project-MetaEQL

Run ```eql_maml.py``` for multi-task experiments, which supports joint training and MAML optimization.

```feynman_ai_equations.py``` sets up the train and test functions - there are some hardcoded options, as well as sine and exponential distributions.

Optimization code is in ```eql_maml.py```, meta-learning utils in ```l2l.py```, and most interesting EQL implementation stuff in the other utils files.
