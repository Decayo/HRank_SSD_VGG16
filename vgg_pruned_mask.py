
import torch
import numpy as np

import pickle


class mask_vgg_16:
    def __init__(self, model=None, compress_rate=[0.50], job_dir='',device=None):
        self.model = model
        self.compress_rate = compress_rate
        self.mask = {}
        self.job_dir=job_dir
        self.device = device
        self.param_per_cov = None
    def layer_mask(self, cov_id, resume=None, param_per_cov=2,  arch="vgg_16_limit5"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume='result\\' + 'mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):
           # print("now index")
           # print(index)
            if index == cov_id * param_per_cov:
                break
            if index == (cov_id - 1) * param_per_cov:
                print("initial ?")
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
               # print(rank)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
               # print(pruned_num)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]
               # print(self.mask[index].shape)
               # print(item.data.shape)
            if index > (cov_id - 1) * param_per_cov and index <= (cov_id - 1) * param_per_cov + param_per_cov-1:
               # print(item.data.shape)
               # print(zeros.shape)
                self.mask[index] = torch.squeeze(zeros)
               # print(self.mask[index].shape)
                #print(item.data)
                #print(self.mask[index])
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            item.data = item.data * self.mask[index]#prune certain weight

