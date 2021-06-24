import argparse
import os
import logging
import sys
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import utils.common as utils
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.config import vgg_ssd_config
import torchvision
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet','mobilenet_v2','mobilenet_v1'),
    help='The architecture to prune')
parser.add_argument('--datasets', nargs='+', help='Dataset directory path')

parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
# Params for SGD
# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--num_workers', default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--batch_size', default=64, type=int,
                    help='Batch size for training')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')
parser.add_argument(
    '--limit',
    type=int,
    default=5,
    help='The num of batch to get rank.')
parser.add_argument(
    '--pretrain_dir',
    type=str,
    default='models/vgg16-ssd-mp-0_7726.pth',
    help='load the model from the specified checkpoint')
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
parser.add_argument('--log_file' , type = str , default = '0530_02')
args = parser.parse_args()
import logging
logging.basicConfig(level=logging.DEBUG, filename='myLog_vgg16_'+args.log_file+'.log', filemode='w')
device = torch.device("cuda:0" if torch.cuda.is_available()
                      and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True
def get_feature_hook(self, input, output):
    logging.info("=========================Start Rank Decomposiotion=============================")
    logging.info('outpus shape')
    logging.info(output.shape)
    global feature_result
    global entropy
    global total
    a = output.shape[0]
    b = output.shape[1]
    logging.info('ab shape')
    logging.info(a)
    logging.info(b)
    logging.info("matrix :")
    cnt_0 = 0
    cnt_10 = 0
    cnt_20 = 0
    cnt_30 = 0
    cnt_31 = 0
    avg_rank = 0
    for i in range(a):
        for j in range(b):
            tmp_r = torch.matrix_rank(output[i,j,:,:]).item()
            avg_rank += tmp_r
            if tmp_r == 0 :
                cnt_0+=1
            elif tmp_r <= 10 :
                cnt_10+=1
            elif tmp_r <= 20 :
                cnt_20+=1
            elif tmp_r <= 30 :
                cnt_30+=1
            else:
                cnt_31+=1
    logging.info("0~30:(%d,%d,%d,%d.%d), avg_rank = %d",cnt_0,cnt_10 , cnt_20 ,cnt_30 ,cnt_31 , avg_rank/(a*b))
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b)])
    c = c.view(a, -1).float()
    #logging.info(c.shape)
    #logging.info(c)
    c = c.sum(0)
    #logging.info(c.shape)
    #logging.info(c)
    feature_result = feature_result * total + c
    #logging.info(feature_result)
    total = total + a
    #logging.info(total)
    feature_result = feature_result / total
    logging.info(feature_result)
    logging.info("==========================End=============================")

def test():
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    limit = args.limit
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            if batch_idx >= limit:  # use the first 6 batches to estimate the rank.
                    break
            images, boxes, labels = data
            images = images.to(device)
            boxes = boxes.to(device)
            labels = labels.to(device)
            num += 1

            
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

            running_loss += loss.item()
            running_regression_loss += regression_loss.item()
            running_classification_loss += classification_loss.item()
            utils.progress_bar(batch_idx, limit, 'val Loss: %.3f ,val_regression_loss %.3f, val_classification_loss %.3f '
                        % (running_loss / num,running_regression_loss / num,running_classification_loss / num))#'''
if __name__ == '__main__':
    timer = Timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logging.info(args)
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
   
    logging.info("Prepare training datasets.")
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)
    datasets = []
    for dataset_path in args.datasets:
        dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
        label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
        store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)
        datasets.append(dataset)
    

    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    logging.info("Build network.")
    net = create_net(num_classes,compress_rate = [0.]*100)
    #net = eval(args.arch)(compress_rate=[0.]*100)
    #_net = create_net(num_classes)
    timer.start("Load Model")

    print(net)

    if len(args.gpu)>1 and torch.cuda.is_available():
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        net = torch.nn.DataParallel(net, device_ids=device_id)
    if args.pretrain_dir:
        
        checkpoint = torch.load(args.pretrain_dir)
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        net.load_state_dict(checkpoint,False)
    else:
        print('please speicify a pretrain model ')
        raise NotImplementedError
    

    net.to(device)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=device)
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    if args.arch=='vgg_16_bn':
    
        if len(args.gpu) > 1:
           #vgg-16 bn relucfg =  [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
            #relucfg = [1,4,6,9,11,13,16,18,20,22,23,25,27,30,32,34]
            relucfg = net.module.relucfg
        else:
            #relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
            relucfg = net.relucfg

        for i, cov_id in enumerate(relucfg):
            logging.info('now conv  : ')
            logging.info(cov_id)
            logging.info(net.base_net[cov_id])
            cov_layer = net.base_net[cov_id]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()

            if not os.path.isdir('rank_conv/'+args.arch+'_limit%d'%(args.limit)):
                os.mkdir('rank_conv/'+args.arch+'_limit%d'%(args.limit))
            np.save('rank_conv/'+args.arch+'_limit%d'%(args.limit)+'/rank_conv' + str(i + 1) + '.npy', feature_result.numpy())

            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

