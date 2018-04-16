from __future__ import print_function
from six.moves import range

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from PIL import Image

from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from model import G_DCGAN, G_NET
from datasets import prepare_data
from model import RNN_ENCODER, CNN_ENCODER

from miscc.losses import words_loss
from miscc.losses import discriminator_loss, generator_loss, KL_loss
import os
import os.path as osp
import time
import numpy as np
import sys
import random

# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, n_words):
        torch.cuda.set_device(cfg.GPU_ID)
        cudnn.benchmark = True

        self.n_words = n_words

        self.noise = Variable(torch.FloatTensor(cfg.OUTPUT_NUM, cfg.GAN.Z_DIM), volatile=True)
        self.noise = self.noise.cuda()
        self.noise.data.normal_(0, 1)

        # Build and load the generator
        self.text_encoder = RNN_ENCODER(self.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
        state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
        self.text_encoder.load_state_dict(state_dict)
        self.text_encoder = self.text_encoder.cuda()
        self.text_encoder.eval()

        self.netG = G_NET()
        s_tmp = cfg.TRAIN.NET_G[:cfg.TRAIN.NET_G.rfind('.pth')]
        model_dir = cfg.TRAIN.NET_G
        state_dict = torch.load(model_dir, map_location=lambda storage, loc: storage)
        self.netG.load_state_dict(state_dict)
        self.netG.cuda()
        self.netG.eval()

        indices = Variable(torch.LongTensor(cfg.SELECTED_INDEX)).cuda()
        self.noise = torch.index_select(self.noise,0,indices)
        cfg.OUTPUT_NUM=len(cfg.SELECTED_INDEX)


    def gen_example(self, data_dic):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            captions, cap_lens, sorted_indices = data_dic

            captions = Variable(torch.from_numpy(captions), volatile=True)
            cap_lens = Variable(torch.from_numpy(cap_lens), volatile=True)
            captions = captions.expand(cfg.OUTPUT_NUM,captions.size(1))
            cap_lens = cap_lens.expand(cfg.OUTPUT_NUM)
            # batch_size = captions.size(0)
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()

            hidden = self.text_encoder.init_hidden(cfg.OUTPUT_NUM)
            words_embs, sent_emb = self.text_encoder(captions, cap_lens, hidden)
            mask = (captions == 0)

            if not cfg.FIX_NOISE:
                self.noise = Variable(torch.FloatTensor(cfg.OUTPUT_NUM, cfg.GAN.Z_DIM), volatile=True)
                self.noise = self.noise.cuda()
                self.noise.data.normal_(0, 1)

            fake_imgs, attention_maps, _, _ = self.netG(self.noise, sent_emb, words_embs, mask)
            # G attention
            cap_lens_np = cap_lens.cpu().data.numpy()
            index = np.arange(cfg.OUTPUT_NUM)
            np.random.shuffle(index)
            for j in range(cfg.OUTPUT_NUM):
                save_name = osp.join(cfg.SAVE_DIR, 'gan_%d.png' % (j))
                im = fake_imgs[2][index[j]].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                # TODO random flip
                im = Image.fromarray(im)
                if random.choice([True, False]):
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                im.save(save_name)
