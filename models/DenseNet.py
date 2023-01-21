#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.densenet121(num_classes=nOut)
