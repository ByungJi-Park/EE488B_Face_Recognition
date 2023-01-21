#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.regnet_y_400mf(num_classes=nOut)
