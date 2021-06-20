import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
#from ..nn.vgg import vgg

from .ssd import SSD
from .predictor import Predictor
from .config import vgg_ssd_config as config


def create_vgg_ssd(num_classes, is_test=False, compress_rate = None):
    vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                  512, 512, 512]
    #print(compress_rate)
    
    base_net = ModuleList(vgg(vgg_config,compress_rate = compress_rate))
    compress_rate = compress_rate
    
    source_layer_indexes = [
        (23, BatchNorm2d(int(512*(1-compress_rate[10])))),
        len(base_net),
    ]
    extras = ModuleList([
        Sequential(
            Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        ),
        Sequential(
            Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            ReLU()
        )
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=int(512*(1-compress_rate[15])), out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=int(512*(1-compress_rate[15])), out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config , compress_rate = compress_rate)


def create_vgg_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor


import torch.nn as nn

def vgg(cfg, batch_norm=False,compress_rate=None):
    layers = []
    in_channels = 3
    cnt = 0
    compress_rate = compress_rate
    #print(f"vgg compress:{compress_rate}")
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            v = int(v * (1-compress_rate[cnt]))
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d.cp_rate = compress_rate[cnt]
            cnt += 1
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    conv6 = nn.Conv2d(in_channels, int(1024*(1-compress_rate[14])), kernel_size=3, padding=6, dilation=6)
    conv6.cp_rate = compress_rate[14]
    conv7 = nn.Conv2d(int(1024*(1-compress_rate[14])), 1024, kernel_size=1)
    #conv7.cp_rate = compress_rate[14]
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers