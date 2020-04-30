This repository contains the PyTorch implementation of the paper [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)

## Network Configurations

#### Train the Resnet110 with stochastic depth on CIFAR-10 : 

```
python main.py --data-root /PATH/TO/CIFAR10 --data cifar10 --save /PATH/TO/SAVE  \
               --pL 0.5 --blocks 18
```

#### Train the Resnet110 with stochastic depth on CIFAR-100 : 

```
python main.py --data-root /PATH/TO/CIFAR100 --data cifar100 --save /PATH/TO/SAVE  \
               --pL 0.5 --blocks 18
```


