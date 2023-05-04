"""

Author: Andreas Rössler
"""
import os
import argparse
import timm

import torch
import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
from network.xception import xception
import math
import torchvision

def get_custom_model(model_name):
    model = None
    linear_params_list = None
    
    if model_name == 'efficientnet_v2_l':
        model = torchvision.models.efficientnet_v2_l(weights=torchvision.models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features=1280, out_features=1, bias=True)  
        )
        linear_params_list = ['classifier.1.weight', 'classifier.1.bias']
    elif model_name == 'convnext_base_in22ft1k':
        model = timm.create_model('convnext_base_in22ft1k', pretrained=True, num_classes=1, drop_rate=0.2)
        linear_params_list = ['head.norm.weight', 'head.norm.bias', 'head.fc.weight', 'head.fc.bias']
    elif model_name == 'swinv2_base_window12to16_192to256_22kft1k':
        model = timm.create_model('swinv2_base_window12to16_192to256_22kft1k', pretrained=True, num_classes=1)
        model.head = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1024, out_features=1, bias=True)
        )
        linear_params_list = ['head.1.weight', 'head.1.bias']
    else:
        pass
        # model = timm.create_model('beit_large_patch16_224', pretrained=True, num_classes=2)
    
    return model, linear_params_list

class SubmissionModel(nn.Module):
    def __init__(self, m1=None, m2=None, m3=None):
        super(SubmissionModel, self).__init__()
        if m3 is not None:
            self.m1 = m1
            # self.m2 = m2
            # self.m3 = m3
        else:
            self.m1, _ = get_custom_model('efficientnet_v2_l')
            # self.m2, _ = get_custom_model('swinv2_base_window12to16_192to256_22kft1k')
            # self.m3, _ = get_custom_model('convnext_base_in22ft1k')
        self.sigmoid = nn.Sigmoid()
        self.avgpool1d = nn.AvgPool1d(3)
        
    def forward(self, x):
        # y1 = self.m1(x)
        # y1 = self.sigmoid(y1)
        # y2 = self.m2(x)
        # y2 = self.sigmoid(y2)
        # y3 = self.m3(x)
        # y3 = self.sigmoid(y3)
        # z = torch.cat((y1, y2, y3), dim=1)
        # y = self.avgpool1d(z)
        y1 = self.m1(x)
        y1 = self.sigmoid(y1)
        # y2 = self.m2(x)
        # y2 = self.sigmoid(y2)
        # y3 = self.m3(x)
        # y3 = self.sigmoid(y3)
        # z = torch.cat((y1, y2, y3), dim=1)
        # y = self.avgpool1d(y1)
        z = torch.cat((1-y1, y1), dim=1)
        # print(y1)
        return z

def return_pytorch04_xception(pretrained=True):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    print("=======WHATS PRETRAINED WANT TRUE:======== ",pretrained)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            '/Users/rishithgandham/Downloads/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model

def effic(pretrained=True):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained=False)
    #print("=======WHATS PRETRAINED WANT TRUE:======== ",pretrained)
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        state_dict = torch.load(
            '/Users/rishithgandham/Downloads/xception-b5690688.pth')
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        model.last_linear = model.fc
        del model.fc
    return model


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet18' or modelchoice == 'efficientnetv2':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=True)
            if modelchoice == 'resnet18':
                self.model = torchvision.models.resnet18(pretrained=True)
            if modelchoice == 'efficientnetv2':
                self.model = SubmissionModel()

                state_dict = self.model.load_state_dict(torch.load('/Users/rishithgandham/Downloads/m1.pt', map_location=torch.device('cpu')), strict=False)
                # state_dict = torch.load('/Users/rishithgandham/Downloads/m1.pt', map_location=torch.device('cpu'))
                # remove_prefix = 'm1.'
                # state_dict = {remove_prefix+k: v for k, v in state_dict.items()}
                # # self.model = self.model.to('cpu')
                # self.model.load_state_dict(state_dict)

            
            # Replace fc
            # num_ftrs = self.model.fc.in_features
            # if not dropout:
            #     self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            # else:
            #     self.model.fc = nn.Sequential(
            #         nn.Dropout(p=dropout),
            #         nn.Linear(num_ftrs, num_out_classes)
            #     )
            # print(self.model)

        else:
            raise Exception('Choose valid model, e.g. resnet50')
        

    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    print("inside model.py file")
    print("you have selected ",modelname)
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes), 299, \
               True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    elif modelname == 'efficientnetv2':
        return TransferModel(modelchoice='efficientnetv2', dropout=dropout,
                             num_out_classes=num_out_classes), \
               224, True, ['image'], None
    
    else:
        raise NotImplementedError(modelname)


if __name__ == '__main__':
    model, image_size, *_ = model_selection('resnet18', num_out_classes=2)
    #print('printing the model in model.py:',model)
    model = model.cuda()
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s))
