import torch.nn as nn


# borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
# source code 沒有 compress_rate 注意
# cnt cov2d cp_rate 都是新加的 
# 
def vgg(cfg, batch_norm=False,compress_rate=None):
    layers = []
    in_channels = 3
    cnt = 0
    compress_rate = compress_rate
    print(compress_rate)
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d.cp_rate = compress_rate[cnt]
            cnt += 1
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers