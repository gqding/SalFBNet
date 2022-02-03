from PIL import Image
import math
import torch
from torch.nn import utils, functional as F
from torch.optim import Adam, SGD
from torch.backends import cudnn
from torchvision import transforms
from torch import nn
from networks.FeedBackNet import FBNet
from networks.FeedBackNet_Res18Fixed import FBNet as FBNet_Res18Fixed
import FBLoss
import numpy as np
import cv2, os, glob
from collections import OrderedDict


class Solver(object):
    def __init__(self, train_loader, val_loader, test_dataset, config):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_dataset = test_dataset
        self.config = config
        # self.writer = SummaryWriter("%s/logs/Summary" % config.save_fold)
        self.beta = math.sqrt(0.3)  # for max F_beta metric
        # inference: choose the side map (see paper)
        self.select = [1, 2, 3, 6]
        self.device = torch.device('cpu')
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        if self.config.cuda:
            cudnn.benchmark = True
            self.device = torch.device('cuda:0')
        self.build_model()
        if self.config.pre_trained:
            best_model=torch.load(self.config.pre_trained)
            new_state_dict = self.check_keys(best_model=best_model)
            self.net.load_state_dict(new_state_dict)
            print('Successfully load the the model from: ' + self.config.pre_trained)
        if config.mode == 'train':
            self.log_output = open("%s/logs/log.txt" % config.save_fold, 'w')
        else:
            best_model=torch.load(self.config.model)
            new_state_dict=self.check_keys(best_model=best_model)
            self.net.load_state_dict(new_state_dict)
            print('Successfully load the the model from: ' + self.config.model)
            
            # self.test_output = open("%s/test.txt" % config.test_fold, 'w')
            self.transform = transforms.Compose([
                # transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def check_keys(self, best_model):
        new_state_dict = OrderedDict()
        state_dict = best_model
        if 'module.' in list(self.net.state_dict().keys())[0] and 'module' not in list(best_model.keys())[
            0]:
            print("checking the keys of the network: network with multi-GPU, but pre-trained model with single-GPU")
            for k, v in state_dict.items():
                name = 'module.' + k
                new_state_dict[name] = v
        elif 'module.' not in list(self.net.state_dict().keys())[0] and 'module' in \
                list(best_model.keys())[0]:
            print("checking the keys of the network: network with multi-GPU, but pre-trained model with single-GPU")
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
        else:
            print("checking the keys of the network: network and pre-trained with multi-GPU, or network and pre-trained with single-GPU")
            new_state_dict = state_dict
        return new_state_dict

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            if p.requires_grad: num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def init_pretrained_weights(self, model, pretrain=True, name='resnet18'):
        if pretrain:
            model_urls = {
                'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
                'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
                'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
                'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
                'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
                'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
                'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
                'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
                'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
            }
            state_dict = torch.hub.load_state_dict_from_url(model_urls[name],
                                                            progress=True)
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys
            pretrained_state_dict = {k: v for k, v in state_dict.items() if k in model_dict.keys()}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_state_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            print('Successfully loaded the model from ' + model_urls[name])
        else:
            self.net.apply(self.net.weights_init)

    # build the network
    def build_model(self):
        if self.config.backbone == 'Res18':
            self.net = FBNet(input_dim=3).to(self.device)
        elif self.config.backbone == 'Res18Fixed':
            self.net = FBNet_Res18Fixed(input_dim=3).to(self.device)
        else:
            raise ModuleNotFoundError
        self.net.apply(self.net.weights_init)
        #self.init_pretrained_weights(model=self.net, pretrain=True, name='resnet50')  # load the part of the pretrained weights
        torch.nn.init.constant_(self.net.c_score_final.weight, 1.0 / 5.0)
        torch.nn.init.constant_(self.net.score_smoothing.weight, 1.0 / (41.0*41.0))
        if torch.cuda.device_count() > 1 and self.config.multi_gpu:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            #dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            self.net = nn.DataParallel(self.net)        
        # if self.config.mode == 'train': self.loss = Loss().to(self.device)
        if self.config.mode == 'train': self.loss = FBLoss.Loss().to(self.device)
        self.net.train()

        #torch.nn.init.constant_(self.net.c_score_final.weight, 1.0 / 5.0)
        # if self.config.load == '': self.net.base.load_state_dict(torch.load(self.config.vgg))
        if self.config.load != '': self.net.load_state_dict(torch.load(self.config.load))
        # self.optimizer = Adam(self.net.parameters(), self.config.lr)
        self.optimizer = SGD(self.net.parameters(), lr=self.config.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        self.print_network(self.net, 'FBNet')

    # update the learning rate
    def update_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return param_group['lr']

    def adjust_learning_rate(self, optimizer, epoch):
        """LR schedule that should yield 76% converged accuracy with batch size 256"""

        if epoch > 1 and epoch % 5 == 0:
            lr = self.config.lr * 0.1
        else:
            lr = self.config.lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def test(self):
        is_show = False
        with torch.no_grad():
            for i, img in enumerate(self.test_dataset):
                img_path = self.test_dataset.image_path[i]
                img_name = os.path.basename(img_path)
                src_image=Image.open(img_path)  # [1024, 768, 3]
                images = self.transform(img).unsqueeze(0)
                # labels = labels.unsqueeze(0)
                shape = (src_image.size[1],src_image.size[0])
                images = images.to(self.device)
                prob_pred, score_list = self.net(images)
                prob_pred = self.normalize_saliency(prob_pred)
                prob_pred = F.interpolate(prob_pred, size=shape, mode='bilinear').cpu().data

                prob_pred_numpy = np.squeeze(prob_pred.cpu().detach().numpy())
                prob_pred_numpy = np.uint8((prob_pred_numpy * 255))
                print('Testing image: '+img_path)

                save_pred_name = os.path.join(self.config.test_fold, img_name.split('.')[0]+'.png')
                cv2.imwrite(save_pred_name, prob_pred_numpy)

                if is_show:
                    import matplotlib.pyplot as plt
                    plt.imshow(prob_pred_numpy)
                    plt.show()

    # training phase
    def train(self):
        pass

    # select best model based on the loss in validation set
    def train_select_best_model(self):
        pass
            

    def normalize_saliency(self, pred_seq):
        sal = torch.reshape(pred_seq, (pred_seq.shape[0], pred_seq.shape[1] * pred_seq.shape[2] * pred_seq.shape[3]))
        max_sal = torch.max(sal, dim=1)[0]
        sal = sal / torch.reshape(max_sal, (sal.shape[0], 1))
        sal = torch.reshape(sal, (pred_seq.shape[0], pred_seq.shape[1], pred_seq.shape[2], pred_seq.shape[3]))
        return sal
