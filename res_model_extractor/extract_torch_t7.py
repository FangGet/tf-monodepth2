import os
import numpy as np
import pickle
import torchfile  # pip install torchfile

# FLAGS(?)
T7_PATH = './resnet-18.t7'

# Open ResNet-18 torch checkpoint
print('Open ResNet-18 torch checkpoint: %s' % T7_PATH)
o = torchfile.load(T7_PATH)

# Load weights in a brute-force way
print('Load weights in a brute-force way')
conv1_weights = o.modules[0].weight
conv1_bn_gamma = o.modules[1].weight
conv1_bn_beta = o.modules[1].bias
conv1_bn_mean = o.modules[1].running_mean
conv1_bn_var = o.modules[1].running_var

conv2_1_weights_1  = o.modules[4].modules[0].modules[0].modules[0].modules[0].weight
conv2_1_bn_1_gamma = o.modules[4].modules[0].modules[0].modules[0].modules[1].weight
conv2_1_bn_1_beta  = o.modules[4].modules[0].modules[0].modules[0].modules[1].bias
conv2_1_bn_1_mean  = o.modules[4].modules[0].modules[0].modules[0].modules[1].running_mean
conv2_1_bn_1_var   = o.modules[4].modules[0].modules[0].modules[0].modules[1].running_var
conv2_1_weights_2  = o.modules[4].modules[0].modules[0].modules[0].modules[3].weight
conv2_1_bn_2_gamma = o.modules[4].modules[0].modules[0].modules[0].modules[4].weight
conv2_1_bn_2_beta  = o.modules[4].modules[0].modules[0].modules[0].modules[4].bias
conv2_1_bn_2_mean  = o.modules[4].modules[0].modules[0].modules[0].modules[4].running_mean
conv2_1_bn_2_var   = o.modules[4].modules[0].modules[0].modules[0].modules[4].running_var
conv2_2_weights_1  = o.modules[4].modules[1].modules[0].modules[0].modules[0].weight
conv2_2_bn_1_gamma = o.modules[4].modules[1].modules[0].modules[0].modules[1].weight
conv2_2_bn_1_beta  = o.modules[4].modules[1].modules[0].modules[0].modules[1].bias
conv2_2_bn_1_mean  = o.modules[4].modules[1].modules[0].modules[0].modules[1].running_mean
conv2_2_bn_1_var   = o.modules[4].modules[1].modules[0].modules[0].modules[1].running_var
conv2_2_weights_2  = o.modules[4].modules[1].modules[0].modules[0].modules[3].weight
conv2_2_bn_2_gamma = o.modules[4].modules[1].modules[0].modules[0].modules[4].weight
conv2_2_bn_2_beta  = o.modules[4].modules[1].modules[0].modules[0].modules[4].bias
conv2_2_bn_2_mean  = o.modules[4].modules[1].modules[0].modules[0].modules[4].running_mean
conv2_2_bn_2_var   = o.modules[4].modules[1].modules[0].modules[0].modules[4].running_var

conv3_1_weights_skip = o.modules[5].modules[0].modules[0].modules[1].weight
conv3_1_weights_1  = o.modules[5].modules[0].modules[0].modules[0].modules[0].weight
conv3_1_bn_1_gamma = o.modules[5].modules[0].modules[0].modules[0].modules[1].weight
conv3_1_bn_1_beta  = o.modules[5].modules[0].modules[0].modules[0].modules[1].bias
conv3_1_bn_1_mean  = o.modules[5].modules[0].modules[0].modules[0].modules[1].running_mean
conv3_1_bn_1_var   = o.modules[5].modules[0].modules[0].modules[0].modules[1].running_var
conv3_1_weights_2  = o.modules[5].modules[0].modules[0].modules[0].modules[3].weight
conv3_1_bn_2_gamma = o.modules[5].modules[0].modules[0].modules[0].modules[4].weight
conv3_1_bn_2_beta  = o.modules[5].modules[0].modules[0].modules[0].modules[4].bias
conv3_1_bn_2_mean  = o.modules[5].modules[0].modules[0].modules[0].modules[4].running_mean
conv3_1_bn_2_var   = o.modules[5].modules[0].modules[0].modules[0].modules[4].running_var
conv3_2_weights_1  = o.modules[5].modules[1].modules[0].modules[0].modules[0].weight
conv3_2_bn_1_gamma = o.modules[5].modules[1].modules[0].modules[0].modules[1].weight
conv3_2_bn_1_beta  = o.modules[5].modules[1].modules[0].modules[0].modules[1].bias
conv3_2_bn_1_mean  = o.modules[5].modules[1].modules[0].modules[0].modules[1].running_mean
conv3_2_bn_1_var   = o.modules[5].modules[1].modules[0].modules[0].modules[1].running_var
conv3_2_weights_2  = o.modules[5].modules[1].modules[0].modules[0].modules[3].weight
conv3_2_bn_2_gamma = o.modules[5].modules[1].modules[0].modules[0].modules[4].weight
conv3_2_bn_2_beta  = o.modules[5].modules[1].modules[0].modules[0].modules[4].bias
conv3_2_bn_2_mean  = o.modules[5].modules[1].modules[0].modules[0].modules[4].running_mean
conv3_2_bn_2_var   = o.modules[5].modules[1].modules[0].modules[0].modules[4].running_var

conv4_1_weights_skip = o.modules[6].modules[0].modules[0].modules[1].weight
conv4_1_weights_1  = o.modules[6].modules[0].modules[0].modules[0].modules[0].weight
conv4_1_bn_1_gamma = o.modules[6].modules[0].modules[0].modules[0].modules[1].weight
conv4_1_bn_1_beta  = o.modules[6].modules[0].modules[0].modules[0].modules[1].bias
conv4_1_bn_1_mean  = o.modules[6].modules[0].modules[0].modules[0].modules[1].running_mean
conv4_1_bn_1_var   = o.modules[6].modules[0].modules[0].modules[0].modules[1].running_var
conv4_1_weights_2  = o.modules[6].modules[0].modules[0].modules[0].modules[3].weight
conv4_1_bn_2_gamma = o.modules[6].modules[0].modules[0].modules[0].modules[4].weight
conv4_1_bn_2_beta  = o.modules[6].modules[0].modules[0].modules[0].modules[4].bias
conv4_1_bn_2_mean  = o.modules[6].modules[0].modules[0].modules[0].modules[4].running_mean
conv4_1_bn_2_var   = o.modules[6].modules[0].modules[0].modules[0].modules[4].running_var
conv4_2_weights_1  = o.modules[6].modules[1].modules[0].modules[0].modules[0].weight
conv4_2_bn_1_gamma = o.modules[6].modules[1].modules[0].modules[0].modules[1].weight
conv4_2_bn_1_beta  = o.modules[6].modules[1].modules[0].modules[0].modules[1].bias
conv4_2_bn_1_mean  = o.modules[6].modules[1].modules[0].modules[0].modules[1].running_mean
conv4_2_bn_1_var   = o.modules[6].modules[1].modules[0].modules[0].modules[1].running_var
conv4_2_weights_2  = o.modules[6].modules[1].modules[0].modules[0].modules[3].weight
conv4_2_bn_2_gamma = o.modules[6].modules[1].modules[0].modules[0].modules[4].weight
conv4_2_bn_2_beta  = o.modules[6].modules[1].modules[0].modules[0].modules[4].bias
conv4_2_bn_2_mean  = o.modules[6].modules[1].modules[0].modules[0].modules[4].running_mean
conv4_2_bn_2_var   = o.modules[6].modules[1].modules[0].modules[0].modules[4].running_var

conv5_1_weights_skip = o.modules[7].modules[0].modules[0].modules[1].weight
conv5_1_weights_1  = o.modules[7].modules[0].modules[0].modules[0].modules[0].weight
conv5_1_bn_1_gamma = o.modules[7].modules[0].modules[0].modules[0].modules[1].weight
conv5_1_bn_1_beta  = o.modules[7].modules[0].modules[0].modules[0].modules[1].bias
conv5_1_bn_1_mean  = o.modules[7].modules[0].modules[0].modules[0].modules[1].running_mean
conv5_1_bn_1_var   = o.modules[7].modules[0].modules[0].modules[0].modules[1].running_var
conv5_1_weights_2  = o.modules[7].modules[0].modules[0].modules[0].modules[3].weight
conv5_1_bn_2_gamma = o.modules[7].modules[0].modules[0].modules[0].modules[4].weight
conv5_1_bn_2_beta  = o.modules[7].modules[0].modules[0].modules[0].modules[4].bias
conv5_1_bn_2_mean  = o.modules[7].modules[0].modules[0].modules[0].modules[4].running_mean
conv5_1_bn_2_var   = o.modules[7].modules[0].modules[0].modules[0].modules[4].running_var
conv5_2_weights_1  = o.modules[7].modules[1].modules[0].modules[0].modules[0].weight
conv5_2_bn_1_gamma = o.modules[7].modules[1].modules[0].modules[0].modules[1].weight
conv5_2_bn_1_beta  = o.modules[7].modules[1].modules[0].modules[0].modules[1].bias
conv5_2_bn_1_mean  = o.modules[7].modules[1].modules[0].modules[0].modules[1].running_mean
conv5_2_bn_1_var   = o.modules[7].modules[1].modules[0].modules[0].modules[1].running_var
conv5_2_weights_2  = o.modules[7].modules[1].modules[0].modules[0].modules[3].weight
conv5_2_bn_2_gamma = o.modules[7].modules[1].modules[0].modules[0].modules[4].weight
conv5_2_bn_2_beta  = o.modules[7].modules[1].modules[0].modules[0].modules[4].bias
conv5_2_bn_2_mean  = o.modules[7].modules[1].modules[0].modules[0].modules[4].running_mean
conv5_2_bn_2_var   = o.modules[7].modules[1].modules[0].modules[0].modules[4].running_var

model_weights_temp = {
    'monodepth2_model/encoder/conv1/conv1/weights': conv1_weights,
    'monodepth2_model/encoder/conv1/BatchNorm/moving_mean': conv1_bn_mean,
    'monodepth2_model/encoder/conv1/BatchNorm/moving_variance': conv1_bn_var,
    'monodepth2_model/encoder/conv1/BatchNorm/beta': conv1_bn_beta,
    'monodepth2_model/encoder/conv1/BatchNorm/gamma': conv1_bn_gamma,

    'monodepth2_model/encoder/conv2_1/conv1/weights': conv2_1_weights_1,
    'monodepth2_model/encoder/conv2_1/BatchNorm/moving_mean':       conv2_1_bn_1_mean,
    'monodepth2_model/encoder/conv2_1/BatchNorm/moving_variance':    conv2_1_bn_1_var,
    'monodepth2_model/encoder/conv2_1/BatchNorm/beta':     conv2_1_bn_1_beta,
    'monodepth2_model/encoder/conv2_1/BatchNorm/gamma':    conv2_1_bn_1_gamma,
    'monodepth2_model/encoder/conv2_1/conv2/weights': conv2_1_weights_2,
    'monodepth2_model/encoder/conv2_1/BatchNorm_1/moving_mean':       conv2_1_bn_2_mean,
    'monodepth2_model/encoder/conv2_1/BatchNorm_1/moving_variance':    conv2_1_bn_2_var,
    'monodepth2_model/encoder/conv2_1/BatchNorm_1/beta':     conv2_1_bn_2_beta,
    'monodepth2_model/encoder/conv2_1/BatchNorm_1/gamma':    conv2_1_bn_2_gamma,
    'monodepth2_model/encoder/conv2_2/conv1/weights': conv2_2_weights_1,
    'monodepth2_model/encoder/conv2_2/BatchNorm/moving_mean':       conv2_2_bn_1_mean,
    'monodepth2_model/encoder/conv2_2/BatchNorm/moving_variance':    conv2_2_bn_1_var,
    'monodepth2_model/encoder/conv2_2/BatchNorm/beta':     conv2_2_bn_1_beta,
    'monodepth2_model/encoder/conv2_2/BatchNorm/gamma':    conv2_2_bn_1_gamma,
    'monodepth2_model/encoder/conv2_2/conv2/weights': conv2_2_weights_2,
    'monodepth2_model/encoder/conv2_2/BatchNorm_1/moving_mean':       conv2_2_bn_2_mean,
    'monodepth2_model/encoder/conv2_2/BatchNorm_1/moving_variance':    conv2_2_bn_2_var,
    'monodepth2_model/encoder/conv2_2/BatchNorm_1/beta':     conv2_2_bn_2_beta,
    'monodepth2_model/encoder/conv2_2/BatchNorm_1/gamma':    conv2_2_bn_2_gamma,

    'monodepth2_model/encoder/conv3_1/shortcut/weights':  conv3_1_weights_skip,
    'monodepth2_model/encoder/conv3_1/conv1/weights': conv3_1_weights_1,
    'monodepth2_model/encoder/conv3_1/BatchNorm/moving_mean':       conv3_1_bn_1_mean,
    'monodepth2_model/encoder/conv3_1/BatchNorm/moving_variance':    conv3_1_bn_1_var,
    'monodepth2_model/encoder/conv3_1/BatchNorm/beta':     conv3_1_bn_1_beta,
    'monodepth2_model/encoder/conv3_1/BatchNorm/gamma':    conv3_1_bn_1_gamma,
    'monodepth2_model/encoder/conv3_1/conv2/weights': conv3_1_weights_2,
    'monodepth2_model/encoder/conv3_1/BatchNorm_1/moving_mean':       conv3_1_bn_2_mean,
    'monodepth2_model/encoder/conv3_1/BatchNorm_1/moving_variance':    conv3_1_bn_2_var,
    'monodepth2_model/encoder/conv3_1/BatchNorm_1/beta':     conv3_1_bn_2_beta,
    'monodepth2_model/encoder/conv3_1/BatchNorm_1/gamma':    conv3_1_bn_2_gamma,
    'monodepth2_model/encoder/conv3_2/conv1/weights': conv3_2_weights_1,
    'monodepth2_model/encoder/conv3_2/BatchNorm/moving_mean':       conv3_2_bn_1_mean,
    'monodepth2_model/encoder/conv3_2/BatchNorm/moving_variance':    conv3_2_bn_1_var,
    'monodepth2_model/encoder/conv3_2/BatchNorm/beta':     conv3_2_bn_1_beta,
    'monodepth2_model/encoder/conv3_2/BatchNorm/gamma':    conv3_2_bn_1_gamma,
    'monodepth2_model/encoder/conv3_2/conv2/weights': conv3_2_weights_2,
    'monodepth2_model/encoder/conv3_2/BatchNorm_1/moving_mean':       conv3_2_bn_2_mean,
    'monodepth2_model/encoder/conv3_2/BatchNorm_1/moving_variance':    conv3_2_bn_2_var,
    'monodepth2_model/encoder/conv3_2/BatchNorm_1/beta':     conv3_2_bn_2_beta,
    'monodepth2_model/encoder/conv3_2/BatchNorm_1/gamma':    conv3_2_bn_2_gamma,

    'monodepth2_model/encoder/conv4_1/shortcut/weights':  conv4_1_weights_skip,
    'monodepth2_model/encoder/conv4_1/conv1/weights': conv4_1_weights_1,
    'monodepth2_model/encoder/conv4_1/BatchNorm/moving_mean':       conv4_1_bn_1_mean,
    'monodepth2_model/encoder/conv4_1/BatchNorm/moving_variance':    conv4_1_bn_1_var,
    'monodepth2_model/encoder/conv4_1/BatchNorm/beta':     conv4_1_bn_1_beta,
    'monodepth2_model/encoder/conv4_1/BatchNorm/gamma':    conv4_1_bn_1_gamma,
    'monodepth2_model/encoder/conv4_1/conv2/weights': conv4_1_weights_2,
    'monodepth2_model/encoder/conv4_1/BatchNorm_1/moving_mean':       conv4_1_bn_2_mean,
    'monodepth2_model/encoder/conv4_1/BatchNorm_1/moving_variance':    conv4_1_bn_2_var,
    'monodepth2_model/encoder/conv4_1/BatchNorm_1/beta':     conv4_1_bn_2_beta,
    'monodepth2_model/encoder/conv4_1/BatchNorm_1/gamma':    conv4_1_bn_2_gamma,
    'monodepth2_model/encoder/conv4_2/conv1/weights': conv4_2_weights_1,
    'monodepth2_model/encoder/conv4_2/BatchNorm/moving_mean':       conv4_2_bn_1_mean,
    'monodepth2_model/encoder/conv4_2/BatchNorm/moving_variance':    conv4_2_bn_1_var,
    'monodepth2_model/encoder/conv4_2/BatchNorm/beta':     conv4_2_bn_1_beta,
    'monodepth2_model/encoder/conv4_2/BatchNorm/gamma':    conv4_2_bn_1_gamma,
    'monodepth2_model/encoder/conv4_2/conv2/weights': conv4_2_weights_2,
    'monodepth2_model/encoder/conv4_2/BatchNorm_1/moving_mean':       conv4_2_bn_2_mean,
    'monodepth2_model/encoder/conv4_2/BatchNorm_1/moving_variance':    conv4_2_bn_2_var,
    'monodepth2_model/encoder/conv4_2/BatchNorm_1/beta':     conv4_2_bn_2_beta,
    'monodepth2_model/encoder/conv4_2/BatchNorm_1/gamma':    conv4_2_bn_2_gamma,

    'monodepth2_model/encoder/conv5_1/shortcut/weights':  conv5_1_weights_skip,
    'monodepth2_model/encoder/conv5_1/conv1/weights': conv5_1_weights_1,
    'monodepth2_model/encoder/conv5_1/BatchNorm/moving_mean':       conv5_1_bn_1_mean,
    'monodepth2_model/encoder/conv5_1/BatchNorm/moving_variance':    conv5_1_bn_1_var,
    'monodepth2_model/encoder/conv5_1/BatchNorm/beta':     conv5_1_bn_1_beta,
    'monodepth2_model/encoder/conv5_1/BatchNorm/gamma':    conv5_1_bn_1_gamma,
    'monodepth2_model/encoder/conv5_1/conv2/weights': conv5_1_weights_2,
    'monodepth2_model/encoder/conv5_1/BatchNorm_1/moving_mean':       conv5_1_bn_2_mean,
    'monodepth2_model/encoder/conv5_1/BatchNorm_1/moving_variance':    conv5_1_bn_2_var,
    'monodepth2_model/encoder/conv5_1/BatchNorm_1/beta':     conv5_1_bn_2_beta,
    'monodepth2_model/encoder/conv5_1/BatchNorm_1/gamma':    conv5_1_bn_2_gamma,
    'monodepth2_model/encoder/conv5_2/conv1/weights': conv5_2_weights_1,
    'monodepth2_model/encoder/conv5_2/BatchNorm/moving_mean':       conv5_2_bn_1_mean,
    'monodepth2_model/encoder/conv5_2/BatchNorm/moving_variance':    conv5_2_bn_1_var,
    'monodepth2_model/encoder/conv5_2/BatchNorm/beta':     conv5_2_bn_1_beta,
    'monodepth2_model/encoder/conv5_2/BatchNorm/gamma':    conv5_2_bn_1_gamma,
    'monodepth2_model/encoder/conv5_2/conv2/weights': conv5_2_weights_2,
    'monodepth2_model/encoder/conv5_2/BatchNorm_1/moving_mean':       conv5_2_bn_2_mean,
    'monodepth2_model/encoder/conv5_2/BatchNorm_1/moving_variance':    conv5_2_bn_2_var,
    'monodepth2_model/encoder/conv5_2/BatchNorm_1/beta':     conv5_2_bn_2_beta,
    'monodepth2_model/encoder/conv5_2/BatchNorm_1/gamma':    conv5_2_bn_2_gamma,
}

# Transpose conv and fc weights
model_weights = {}
for k, v in model_weights_temp.items():
    if len(v.shape) == 4:
        model_weights[k] = np.transpose(v, (2, 3, 1, 0))
    elif len(v.shape) == 2:
        model_weights[k] = np.transpose(v)
    else:
        model_weights[k] = v

np.save('resnet18.npy',model_weights)