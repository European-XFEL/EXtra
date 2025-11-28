"""
BNN implementation.
"""

from sklearn.base import BaseEstimator, RegressorMixin
from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
from scipy.special import gamma

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader
    from torch.nn.modules.utils import _single, _pair, _triple
except ImportError:
    raise Exception("PyTorch not available.")

class BayesLinearEmpiricalPrior(nn.Module):
    """
    Applies Bayesian Linear

    Args:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    """
    __constants__ = ['prior_mu', 'prior_sigma', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinearEmpiricalPrior, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.prior_log_sigma_w = nn.Parameter(torch.ones((out_features, in_features))*np.log(prior_sigma))
        self.prior_log_sigma_b = nn.Parameter(torch.ones((out_features,))*np.log(prior_sigma))

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)

        if bias is None or bias is False :
            self.bias = False
        else :
            self.bias = True

        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(np.log(self.prior_sigma))
        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(np.log(self.prior_sigma))

    def freeze(self) :
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias :
            self.bias_eps = torch.randn_like(self.bias_log_sigma)

    def unfreeze(self) :
        self.weight_eps = None
        if self.bias :
            self.bias_eps = None

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps

        if self.bias:
            if self.bias_eps is None :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        else :
            bias = None

        return F.linear(input, weight, bias)

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_sigma, self.in_features, self.out_features, self.bias is not None)

class BayesLinear(nn.Module):
    r"""
    Applies Bayesian Linear

    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    """
    __constants__ = ['prior_mu', 'bias', 'in_features', 'out_features']

    def __init__(self, prior_mu, prior_sigma, in_features, out_features, bias=True):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.prior_mu = prior_mu
        self.prior_log_sigma = nn.Parameter(torch.ones(1)*np.log(prior_sigma), requires_grad=True)

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_eps', None)

        if bias is None or bias is False:
            self.bias = False
        else :
            self.bias = True

        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma.detach()[0])
        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma.detach()[0])

    def freeze(self) :
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias :
            self.bias_eps = torch.randn_like(self.bias_log_sigma)

    def unfreeze(self) :
        self.weight_eps = None
        if self.bias:
            self.bias_eps = None

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps

        if self.bias:
            if self.bias_eps is None :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        else :
            bias = None

        return F.linear(input, weight, bias)

    def extra_repr(self):
        r"""
        Overriden.
        """
        return 'prior_mu={}, prior_sigma={}, in_features={}, out_features={}, bias={}'.format(self.prior_mu, self.prior_sigma, self.in_features, self.out_features, self.bias is not None)

class _BayesConvNd(nn.Module):
    r"""
    Applies Bayesian Convolution

    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    .. note:: other arguments are following conv of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
    """
    __constants__ = ['prior_mu', 'prior_sigma', 'stride', 'padding', 'dilation',
                     'groups', 'bias', 'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']

    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_BayesConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.prior_log_sigma = math.log(prior_sigma)
                
        if transposed:
            self.weight_mu = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.weight_log_sigma = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.register_buffer('weight_eps', None)
        else:
            self.weight_mu = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.weight_log_sigma = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.register_buffer('weight_eps', None)
            
        if bias is None or bias is False :
            self.bias = False
        else :
            self.bias = True
        
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_channels))
            self.register_buffer('bias_eps', None)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        # Initialization method of Adv-BNN.
        n = self.in_channels
        n *= self.kernel_size[0] ** 2
        stdv = 1.0 / math.sqrt(n)
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_log_sigma.data.fill_(self.prior_log_sigma)

        if self.bias :
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_log_sigma.data.fill_(self.prior_log_sigma)

        # Initialization method of the original torch nn.conv.
#         init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
#         self.weight_log_sigma.data.fill_(self.prior_log_sigma)
        
#         if self.bias :
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_mu)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias_mu, -bound, bound)
           
#             self.bias_log_sigma.data.fill_(self.prior_log_sigma)

    def freeze(self) :
        self.weight_eps = torch.randn_like(self.weight_log_sigma)
        if self.bias :
            self.bias_eps = torch.randn_like(self.bias_log_sigma)
        
    def unfreeze(self) :
        self.weight_eps = None
        if self.bias :
            self.bias_eps = None 

    def extra_repr(self):
        s = ('{prior_mu}, {prior_sigma}'
             ', {in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is False:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_BayesConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class BayesConv2d(_BayesConvNd):
    r"""
    Applies Bayesian Convolution for 2D inputs

    Arguments:
        prior_mu (Float): mean of prior normal distribution.
        prior_sigma (Float): sigma of prior normal distribution.

    .. note:: other arguments are following conv of pytorch 1.2.0.
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
    
    """
    def __init__(self, prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(BayesConv2d, self).__init__(
            prior_mu, prior_sigma, in_channels, out_channels, kernel_size, stride, 
            padding, dilation, False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        
        if self.bias:
            if self.bias_eps is None :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)
            else :
                bias = self.bias_mu + torch.exp(self.bias_log_sigma) * self.bias_eps
        else :
            bias = None
            
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        r"""
        Overriden.
        """
        if self.weight_eps is None :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_log_sigma)
        else :
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * self.weight_eps
        
        return self.conv2d_forward(input, weight)

def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1) :
    """
    An method for calculating KL divergence between two Normal distribtuion.

    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).

    """
    if isinstance(log_sigma_1, float):
        sigma_1 = np.exp(log_sigma_1)
    else:
        sigma_1 = torch.exp(log_sigma_1)
    kl = log_sigma_1 - log_sigma_0 + \
    (torch.exp(log_sigma_0)**2 + (mu_0-mu_1)**2)/(2*sigma_1**2) - 0.5
    return kl.sum()

class BKLLoss(nn.Module):
    """
    Loss for calculating KL divergence of baysian neural network model.

    Args:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.
    """
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', last_layer_only=False):
        super(BKLLoss, self).__init__()
        self.last_layer_only = last_layer_only
        self.reduction = reduction

    def forward(self, model):
        """
        Args:
            model (nn.Module): a model to be calculated for KL-divergence.
        """
        #return bayesian_kl_loss(model, reduction=self.reduction, last_layer_only=self.last_layer_only)
        device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
        kl = torch.Tensor([0]).to(device)
        kl_sum = torch.Tensor([0]).to(device)
        n = torch.Tensor([0]).to(device)

        for m in model.modules() :
            if isinstance(m, (BayesLinearEmpiricalPrior)):
                kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma_w)
                kl_sum += kl
                n += len(m.weight_mu.view(-1))

                if m.bias :
                    kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma_b)
                    kl_sum += kl
                    n += len(m.bias_mu.view(-1))
            if isinstance(m, (BayesLinear, BayesConv2d)):
                kl = _kl_loss(m.weight_mu, m.weight_log_sigma, m.prior_mu, m.prior_log_sigma)
                kl_sum += kl
                n += len(m.weight_mu.view(-1))

                if m.bias :
                    kl = _kl_loss(m.bias_mu, m.bias_log_sigma, m.prior_mu, m.prior_log_sigma)
                    kl_sum += kl
                    n += len(m.bias_mu.view(-1))

        if self.last_layer_only or n == 0 :
            return kl

        if self.reduction == 'mean':
            return kl_sum/n
        elif self.reduction == 'sum':
            return kl_sum
        else:
            raise ValueError(f"{self.reduction} is not valid")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



class BNN(nn.Module):
    """
        A model Bayesian Neural network.
        Each weight is represented by a Gaussian with a mean and a standard deviation.
        Each evaluation of forward leads to a different choice of the weights, so running
        forward several times we can check the effect of the weights variation on the same input.
        The neg_log_likelihood function implements the negative log likelihood to be used as the first part of the loss
        function (the second shall be the Kullback-Leibler divergence).
        The negative log-likelihood is simply the negative log likelihood of a Gaussian
        between the prediction and the true value. The standard deviation of the Gaussian is left as a
        parameter to be fit: sigma.
    """
    def __init__(self, input_dimension: int=1, output_dimension: int=1, rvm: bool=False):
        super(BNN, self).__init__()
        hidden_dimension = 50
        # controls the aleatoric uncertainty
        self.log_isigma2 = nn.Parameter(-torch.ones(1, output_dimension)*np.log(0.1**2), requires_grad=True)
        # controls the weight hyperprior
        self.log_ilambda2 = -np.log(0.1**2)

        # inverse Gamma hyper prior alpha and beta
        #
        # Hyperprior choice on the weights:
        # We want to allow the hyperprior on the weights' variance to have large variance,
        # so that the weights prior can be anything, if possible, but at the same time prevent it from going to infinity
        # (which would allow the weights to be anything, but remove regularization and de-stabilize the fit).
        # Therefore, the weights should be allowed to have high std. dev. on their priors, just not so much so that the fit is unstable.
        # At the same time, the prior std. dev. should not be too small (that would regularize too much.
        # The values below have been taken from BoTorch (alpha, beta) = (3.0, 6.0) and seem to work well if the inputs have been standardized.
        # They lead to a high mean for the weights std. dev. (18) and a large variance (sqrt(var) = 10.4), so that the weights prior is large
        # and the only regularization is to prevent the weights from becoming > 18 + 3 sqrt(var) ~= 50, making this a very loose regularization.
        # An alternative would be to set the (alpha, beta) both to very low values, whichmakes the hyper prior become closer to the non-informative Jeffrey's prior.
        # Using this alternative (ie: (0.1, 0.1) for the weights' hyper prior) leads to very large lambda and numerical issues with the fit.
        self.alpha_lambda = 0.0001
        self.beta_lambda = 0.0001

        # Hyperprior choice on the likelihood noise level:
        # The likelihood noise level is controlled by sigma in the likelihood and it should be allowed to be very broad, but different
        # from the weights prior, it must be allowed to be small, since if we have a lot of data, it is conceivable that there is little noise in the data.
        # We therefore want to have high variance in the hyperprior for sigma, but we do not need to prevent it from becoming small.
        # Making both alpha and beta small makes the gamma distribution closer to the Jeffey's prior, which makes it non-informative
        # This seems to lead to a larger training time, though.
        # Since, after standardization, we know to expect the variance to be of order (1), we can select also alpha and beta leading to high variance in this range
        self.alpha_sigma = 0.0001
        self.beta_sigma = 0.0001

        if rvm:
            self.model = nn.Sequential(
                                       BayesLinearEmpiricalPrior(prior_mu=0.0,
                                                                     prior_sigma=np.exp(-0.5*self.log_ilambda2),
                                                                     in_features=input_dimension,
                                                                     out_features=hidden_dimension),
                                       nn.ReLU(),
                                       BayesLinearEmpiricalPrior(prior_mu=0.0,
                                                                     prior_sigma=np.exp(-0.5*self.log_ilambda2),
                                                                     in_features=hidden_dimension,
                                                                     out_features=output_dimension)
                                        )
        else:
            self.model = nn.Sequential(
                                       BayesConv2d(prior_mu=0.0,
                                                   prior_sigma=np.exp(-0.5*self.log_ilambda2),
                                                   in_channels=input_dimension,
                                                   out_channels=hidden_dimension,
                                                   kernel_size=15, padding=7,
                                                   ),
                                       BayesConv2d(prior_mu=0.0,
                                                   prior_sigma=np.exp(-0.5*self.log_ilambda2),
                                                   in_channels=hidden_dimension,
                                                   out_channels=output_dimension,
                                                   kernel_size=1,
                                                   )
                                       nn.ReLU(),
                                        )
        self.rvm = rvm

    def prune(self):
        """Prune weights."""
        with torch.no_grad():
            for layer in self.model.modules():
                if isinstance(layer, BayesLinearEmpiricalPrior):
                    log_isigma2 = -2.0*layer.prior_log_sigma_w
                    isigma2 = torch.exp(log_isigma2)
                    keep = isigma2 < 1e4
                    layer.weight_mu[~keep] *= 0.0
                    layer.weight_log_sigma[~keep] = -12.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the result f(x) applied on the input x.
        """
        return self.model(x)

    def neg_log_gamma(self, log_x: torch.Tensor, x: torch.Tensor, alpha, beta) -> torch.Tensor:
        """
        Return the negative log of the gamma pdf.
        """
        return -alpha*np.log(beta) - (alpha - 1)*log_x + beta*x + gamma(alpha)

    def neg_log_likelihood(self, prediction: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative log-likelihood (divided by the batch size, since we take the mean).
        """
        error = w*(prediction - target)
        squared_error = error**2
        sigma2 = torch.exp(-self.log_isigma2)
        norm_error = 0.5*squared_error/sigma2
        norm_term = 0.5*(np.log(2*np.pi) - self.log_isigma2)
        return (norm_error + norm_term).sum(dim=1).mean(dim=0)

    def neg_log_hyperprior(self) -> torch.Tensor:
        """
        Calculate the negative log of the hyperpriors.
        """
        # hyperprior for sigma to avoid large or too small sigma
        # with a standardized input, this hyperprior forces sigma to be
        # on avg. 1 and it is broad enough to allow for different sigma
        isigma2 = torch.exp(self.log_isigma2)
        neg_log_hyperprior_noise = self.neg_log_gamma(self.log_isigma2, isigma2, self.alpha_sigma, self.beta_sigma).sum()
        if self.rvm:
            log_ilambda2 = [-2.0*self.model[0].prior_log_sigma_w,
                            -2.0*self.model[2].prior_log_sigma_w,
                            -2.0*self.model[0].prior_log_sigma_b,
                            -2.0*self.model[2].prior_log_sigma_b
                            ]
        else:
            log_ilambda2 = [-2.0*self.model[0].prior_log_sigma,
                            -2.0*self.model[2].prior_log_sigma,
                            ]
        ilambda2 = [torch.exp(k) for k in log_ilambda2]
        neg_log_hyperprior_weights = sum(self.neg_log_gamma(log_k, k, self.alpha_lambda, self.beta_lambda).sum()
                                         for log_k, k in zip(log_ilambda2, ilambda2))
        return neg_log_hyperprior_noise + neg_log_hyperprior_weights

    def aleatoric_uncertainty(self) -> torch.Tensor:
        """
            Get the aleatoric component of the uncertainty.
        """
        #return 0
        return torch.exp(-0.5*self.log_isigma2)

    def w_precision(self) -> torch.Tensor:
        """
            Get the weights precision.
        """
        if self.rvm:
            log_ilambda2 = [-2.0*self.model[0].prior_log_sigma_w,
                            -2.0*self.model[2].prior_log_sigma_w,
                            -2.0*self.model[0].prior_log_sigma_b,
                            -2.0*self.model[2].prior_log_sigma_b
                            ]
        else:
            log_ilambda2 = [-2.0*self.model[0].prior_log_sigma,
                            -2.0*self.model[2].prior_log_sigma,
                            ]
        ilambda2 = [torch.exp(k) for k in log_ilambda2]
        return sum(k.mean() for k in ilambda2)/len(ilambda2)

class BNNModel(RegressorMixin, BaseEstimator):
    """
    Regression model with uncertainties.

    Args:
    """
    def __init__(self, state_dict=None, rvm: bool=False, n_epochs: int=250):
        if state_dict is not None:
            Nx = state_dict["model.0.weight_mu"].shape[1]
            Ny = state_dict["model.2.weight_mu"].shape[0]
            self.model = BNN(Nx, Ny, rvm=rvm)
            self.model.load_state_dict(state_dict)
        else:
            self.model = BNN(rvm=rvm)
        self.rvm = rvm
        self.n_epochs = n_epochs
        self.model.eval()

    def state_dict(self) -> Dict[str, Any]:
        return self.model.state_dict()

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray]=None, **fit_params) -> RegressorMixin:
        """
        Perform the fit and evaluate uncertainties with the test set.

        Args:
          X: The input.
          y: The target.
          weights: The weights.
          fit_params: If it contains X_test and y_test, they are used to validate the model.

        Returns: The object itself.
        """
        if weights is None:
            weights = np.ones(len(X), dtype=np.float32)
        if len(weights.shape) == 1:
            weights = weights[:, np.newaxis]

        ds = TensorDataset(torch.from_numpy(X),
                           torch.from_numpy(y),
                           torch.from_numpy(weights))

        # create model
        self.model = BNN(X.shape[1], y.shape[1], rvm=self.rvm)

        # prepare data loader
        B = 50
        loader = DataLoader(ds,
                            batch_size=B,
                            num_workers=20,
                            shuffle=True,
                            #pin_memory=True,
                            drop_last=True,
                            )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        number_of_batches = len(ds)/float(B)
        weight_prior = 1.0/float(number_of_batches)
        # the NLL is divided by the number of batch samples
        # so divide also the prior losses by the number of batch elements, so that the
        # function optimized is F/# samples
        # https://arxiv.org/pdf/1505.05424.pdf
        weight_prior /= float(B)

        # KL loss
        kl_loss = BKLLoss(reduction='sum', last_layer_only=False)

        # train
        self.model.train()
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
        for epoch in range(self.n_epochs):
            meter = {k: AverageMeter(k, ':6.3f')
                    for k in ('loss', '-log(lkl)', '-log(prior)', '-log(hyper)', 'sigma', 'w.prec.')}
            progress = ProgressMeter(
                            len(loader),
                            meter.values(),
                            prefix="Epoch: [{}]".format(epoch))
            for i, batch in enumerate(loader):
                x_b, y_b, w_b = batch
                if torch.cuda.is_available():
                    x_b = x_b.to('cuda')
                    y_b = y_b.to('cuda')
                    w_b = w_b.to('cuda')
                y_b_pred = self.model(x_b)

                nll = self.model.neg_log_likelihood(y_b_pred, y_b, w_b)
                nlprior = weight_prior * kl_loss(self.model)
                nlhyper = weight_prior * self.model.neg_log_hyperprior()
                loss = nll + nlprior + nlhyper

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                meter['loss'].update(loss.detach().cpu().item(), B)
                meter['-log(lkl)'].update(nll.detach().cpu().item(), B)
                meter['-log(prior)'].update(nlprior.detach().cpu().item(), B)
                meter['-log(hyper)'].update(nlhyper.detach().cpu().item(), B)
                meter['sigma'].update(self.model.aleatoric_uncertainty().mean().detach().cpu().numpy(), B)
                meter['w.prec.'].update(self.model.w_precision().detach().cpu().item(), B)

            progress.display(len(loader))
        if self.rvm:
            self.model.prune()

        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to('cpu')

        return self

    def predict(self, X: np.ndarray, return_std: bool=False) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Predict y from X.

        Args:
          X: Input dataset.

        Returns: Predicted Y and, if return_std is True, also its uncertainty.
        """
        K = 10
        y_pred = list()
        for _ in range(K):
            y_k = self.model(torch.from_numpy(X)).detach().cpu().numpy()
            y_pred.append(y_k)
        y_pred = np.stack(y_pred, axis=1)
        y_mu = np.mean(y_pred, axis=1)
        y_epi = np.std(y_pred, axis=1)
        y_ale = self.model.aleatoric_uncertainty().detach().cpu().numpy()
        y_unc = (y_epi**2 + y_ale**2)**0.5
        if not return_std:
            return y_mu
        return y_mu, y_unc

