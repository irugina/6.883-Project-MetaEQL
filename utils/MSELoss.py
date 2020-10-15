import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.torch_regularization import l12_smooth


class MSELoss(nn.Module):
    def __init__(self, reg_weight):
        super(MSELoss, self).__init__()
        self.reg_weight = reg_weight

    def forward(self, input, target, a=0.05):
        """input: predictions"""

        error = F.mse_loss(input, target)
        # error_test = tf.losses.mean_squared_error(labels=y_test, predictions=y_hat)

        # TODO: make l12_smooth a Module
        # TODO: regularization
        weights = sym.get_weights()
        # reg_loss = l12_smooth()

        if type(weights) == list:
            reg_loss = sum([l12_smooth(tensor) for tensor in weights])
        else:
            smooth_abs = torch.where(torch.abs(weights) < a,
                                     torch.pow(weights, 4) / (-8 * a ** 3) + torch.square(
                                         weights) * 3 / 4 / a + 3 * a / 8,
                                     torch.abs(weights))

            reg_loss = torch.sum(torch.sqrt(smooth_abs))

        loss = error + self.reg_weight * reg_loss
        return loss
