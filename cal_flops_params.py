
import torch
import argparse
import get_flops

from thop import profile
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.config import vgg_ssd_config

parser = argparse.ArgumentParser(description='Calculating flops and params')

parser.add_argument(
    '--input_image_size',
    type=int,
    default=32,
    help='The input_image_size')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16',
    choices=('vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet','resnet_50','mobilenet_v2','mobilenet_v1'),
    help='The architecture to prune')
parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')
args = parser.parse_args()
num_classes = 20
if args.compress_rate:
    import re

    cprate_str = args.compress_rate
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    compress_rate = cprate
create_net = create_vgg_ssd
config = vgg_ssd_config
model = create_net(num_classes,compress_rate=compress_rate)
print('compress rate: ',compress_rate)
model.eval()
device = torch.device("cpu")
# calculate model size
# input_image_size = args.input_image_size
# input_image = torch.randn(1, 3, input_image_size, input_image_size).cuda()
# flops, params = profile(model, inputs=(input_image,))
flops, params= get_flops.measure_model(model,device,3,args.input_image_size,args.input_image_size)
print('Params: %.2f'%(params))
print('Flops: %.2f'%(flops))

