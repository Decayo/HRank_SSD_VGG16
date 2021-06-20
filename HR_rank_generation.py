import argparse
import os
import logging
import sys
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch
from models.mobilenetv2 import mobilenet_v2
from torch.utils.data import DataLoader, ConcatDataset
import utils.common as utils
from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_large_ssd_lite, create_mobilenetv3_small_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

import torchvision
parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument(
    '--arch',
    type=str,
    default='mobilenet_v2',
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet','mobilenet_v2','mobilenet_v1'),
    help='The architecture to prune')
parser.add_argument('--datasets', nargs='+', help='Dataset directory path')

parser.add_argument('--net', default="mb2-ssd-lite",
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
parser.add_argument('--batch_size', default=128, type=int,
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
    default='models/mb2-ssd-lite-mp-0_686.pth',
    help='load the model from the specified checkpoint')
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available()
                      and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True
def get_feature_hook(self, input, output):
    global feature_result
    global entropy
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item()
                     for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total


def inference():
        global best_acc
        net.eval()
        #_net.eval()
        running_loss = 0.0
        running_regression_loss = 0.0
        running_classification_loss = 0.0
        num = 0
        limit = args.limit
        with torch.no_grad():
            for batch_idx,data in enumerate(train_loader):
                # use the first 5 batches to estimate the rank.
                # print("batch_idx : " +  batch_idx)
                
                if batch_idx >= limit:
                    break
                images, boxes, labels = data
                images = images.to(device)
                boxes = boxes.to(device)
                labels = labels.to(device)
                num += 1
                confidence, locations= _net(images)
                outputs = net(images)
                regression_loss, classification_loss = criterion(
                    confidence, locations, labels, boxes)
                loss = regression_loss + classification_loss
                running_loss  += loss.item()
                running_regression_loss += regression_loss.item()
                running_classification_loss += classification_loss.item()
                utils.progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (running_loss/(batch_idx+1), 100.*running_regression_loss/num, running_classification_loss, num))
def test():
    _net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    limit = args.limit
    for batch_idx, data in enumerate(train_loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = _net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        utils.progress_bar(batch_idx, limit, 'val Loss: %.3f val_classification_loss %.3f '
                    % (running_regression_loss / num, running_classification_loss / num))#'''
if __name__ == '__main__':
    timer = Timer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(args)
    if args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num,compress_rate=[0.]*100, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
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
    _net = create_net(num_classes)
    #net = eval(args.arch)(compress_rate=[0.]*100)

    timer.start("Load Model")

    print(_net)

    if len(args.gpu)>1 and torch.cuda.is_available():
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        #net = torch.nn.DataParallel(net, device_ids=device_id)
        _net = torch.nn.DataParallel(_net, device_ids=device_id)
    if args.pretrain_dir:
        
        checkpoint = torch.load(args.pretrain_dir)
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        #net.load_state_dict(checkpoint,False)
        _net.load_state_dict(checkpoint,False)
    else:
        print('please speicify a pretrain model ')
        raise NotImplementedError
    _net.to(device)
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=device)
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    # logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
    #              + f"Extra Layers learning rate: {extra_layers_lr}.")

    # if args.scheduler == 'multi-step':
    #     logging.info("Uses MultiStepLR scheduler.")
    #     milestones = [int(v.strip()) for v in args.milestones.split(",")]
    #     scheduler = MultiStepLR(optimizer, milestones=milestones,
    #                                                  gamma=0.1, last_epoch=last_epoch)
    # elif args.scheduler == 'cosine':
    #     logging.info("Uses CosineAnnealingLR scheduler.")
    #     scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    # else:
    #     logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)

    cov_layer = eval('_net.base_net[0]')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    test()
    handler.remove()
    if not os.path.isdir('rank_conv/' + args.arch+'_limit%d'%(args.limit)):
        os.mkdir('rank_conv/' + args.arch+'_limit%d'%(args.limit))
    np.save('rank_conv/' + args.arch+'_limit%d'%(args.limit)+ '/rank_conv%d' % (1) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)
    cnt=1
    for i in range(1,19):
    
        if i==1:
            block = eval('_net.base_net[%d].conv' % (i))
            relu_list=[2,4]
        elif i==18:
            block = eval('_net.base_net[%d]' % (i))
            relu_list=[2]
        else:
            block = eval('_net.base_net[%d].conv' % (i))
            relu_list = [2,5,7]
        for j in relu_list:
            cov_layer = block[j]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + args.arch +'_limit%d'%(args.limit)+ '/rank_conv%d'%(cnt + 1)+'.npy', feature_result.numpy())
            cnt+=1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)
    
    # logging.info(f"Start training from epoch {last_epoch + 1}.")
    # for epoch in range(last_epoch + 1, args.num_epochs):
    #     scheduler.step()
    #     train(train_loader, net, criterion, optimizer,
    #           device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        
    #     if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
    #         val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
    #         logging.info(
    #             f"Epoch: {epoch}, " +
    #             f"Validation Loss: {val_loss:.4f}, " +
    #             f"Validation Regression Loss {val_regression_loss:.4f}, " +
    #             f"Validation Classification Loss: {val_classification_loss:.4f}"
    #         )
    #         model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
    #         net.save(model_path)
    #         logging.info(f"Saved model {model_path}")
