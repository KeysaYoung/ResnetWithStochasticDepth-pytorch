import os
import glob
import time
import argparse

model_names = ['resnet']

arg_parser = argparse.ArgumentParser(
                description='Resnet With Stochastic Depth')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save', default='save/default-{}'.format(time.time()),
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory'
                       '(default: save/debug)')
exp_group.add_argument('--resume', action='store_true',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu', default=None, type=str, help='GPU available.')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar10',
                        choices=['cifar10', 'cifar100', 'ImageNet'],
                        help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true', default=False,
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

# model arch related
arch_group = arg_parser.add_argument_group('arch',
                                           'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: msdnet)')
# msdnet config
arch_group.add_argument('--nBlocks', type=int, default=1)
arch_group.add_argument('--pL', type=float, default=0.5)
arch_group.add_argument('--blocks', type=int, default=18, 
                        help='number of residual blocks for every group')



# training related
optim_group = arg_parser.add_argument_group('optimization',
                                            'optimization setting')
optim_group.add_argument('-b', '--batch-size', default=128, type=int,
                         metavar='N', help='mini-batch size (default: 64)')
