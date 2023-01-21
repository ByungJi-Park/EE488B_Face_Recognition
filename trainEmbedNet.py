#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import time
import os
import argparse
import pdb
import glob
import datetime
from utils import *
from EmbedNet import *
from DatasetLoader import get_data_loader
import torchvision.transforms as transforms

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "Face Recognition Training");

## Data loader
parser.add_argument('--batch_size',         type=int, default=100,	help='Batch size, number of classes per batch');
parser.add_argument('--max_img_per_cls',    type=int, default=500,	help='Maximum number of images per class per epoch');
parser.add_argument('--nDataLoaderThread',  type=int, default=5, 	help='Number of loader threads');

## Training details
parser.add_argument('--test_interval',  type=int,   default=5,      help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=50,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="softmax",  help='Loss function');
parser.add_argument('--vgg',            type=bool,  default=False, help='Whether using vgg or not')

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.90,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerClass',      type=int,   default=1,      help='Number of images per class per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=9500,   help='Number of classes in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs');

## Training and evaluation data
# for ee488b data
#parser.add_argument('--train_path',     type=str,   default="data/train",   help='Absolute path to the train set');
# for vgg data
parser.add_argument('--train_path',     type=str,   default="data/vgg1",   help='Absolute path to the train set');
parser.add_argument('--train_ext',      type=str,   default="jpg",  help='Training files extension');
parser.add_argument('--test_path',      type=str,   default="data/val",     help='Absolute path to the test set');
parser.add_argument('--test_list',      type=str,   default="data/val_pairs.csv",   help='Evaluation list');

## Model definition
parser.add_argument('--model',          type=str,   default="ResNet18", help='Name of model definition');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true',   help='Eval only')
parser.add_argument('--output',         type=str,   default="",     help='Save a log of output to this file name');

## Training
parser.add_argument('--mixedprec',      dest='mixedprec',   action='store_true', help='Enable mixed precision training')
parser.add_argument('--gpu',            type=int,   default=9,      help='GPU index');

## Fine-tuning
parser.add_argument('--fine-tune',      type=bool,   default=False,      help='Fine tuning');

args = parser.parse_args();

'''
print(vars(args))
{'batch_size': 200, 'max_img_per_cls': 500, 'nDataLoaderThread': 5, 'test_interval': 5, 'max_epoch': 50, 'trainfunc': 'amsoftmax', 
'optimizer': 'adam', 'scheduler': 'steplr', 'lr': 0.001, 'lr_decay': 0.9, 'weight_decay': 0, 'margin': 0.1, 'scale': 30, 'nPerClass': 1, 
'nClasses': 2000, 'initial_model': '', 'save_path': 'exps/exp1', 'train_path': 'data/train', 'train_ext': 'jpg', 'test_path': 'data/val', 
'test_list': 'data/val_pairs.csv', 'model': 'ResNet18', 'nOut': 512, 'eval': False, 'output': '', 'mixedprec': False, 'gpu': 0}
'''

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Trainer script
# ## ===== ===== ===== ===== ===== ===== ===== =====
## Change
def main_worker(args):

    ## Load models
    # **vars
    s = EmbedNet(**vars(args)).cuda();

    it          = 1

    ## Input transformations for training
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.RandomCrop([224,224]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Input transformations for evaluation
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.CenterCrop([224,224]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(transform=train_transform, **vars(args));

    trainer     = ModelTrainer(s, **vars(args))

    ## Load model weights
    modelfiles = glob.glob('{}/model0*.model'.format(args.save_path))
    modelfiles.sort()

    ## Fine tuning
    if args.fine_tune:
        trainer.fine_tuning(args.initial_model);
        print("Model {} loaded!".format(args.initial_model));
    ## If initial_model exists, start from that file
    elif(args.initial_model != ""):
        trainer.loadParameters(args.initial_model);
        print("Model {} loaded!".format(args.initial_model));
    ## If the target directory already exists, start from the existing file
    elif len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1]);
        print("Model {} loaded from previous state!".format(modelfiles[-1]));
        it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1

    ## If the current iteration is not 1, update the scheduler
    for ii in range(1,it):
        trainer.__scheduler__.step()

    ## Print total number of model parameters
    pytorch_total_params = sum(p.numel() for p in s.__S__.parameters())
    print('Total model parameters: {:,}'.format(pytorch_total_params))
    
    ## Evaluation code 
    if args.eval == True:

        sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

        print('EER {:.4f}'.format(result[1]))

        if args.output != '':
            with open(args.output,'w') as f:
                for ii in range(len(sc)):
                    f.write('{:4f},{:d},{}\n'.format(sc[ii],lab[ii],trials[ii]))

        quit();


    ## Write args to scorefile for training
    scorefile = open(args.save_path+"/scores.txt", "a+");

    strtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scorefile.write('{}\n{}\n'.format(strtime,args))
    scorefile.flush()

    ## vgg training script
    if args.vgg == True:
        for it in range(1, args.max_epoch+1):

            clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

            print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training epoch {:d} with LR {:.5f} ".format(it,max(clr)));

            loss = trainer.train_network(trainLoader);

            trainer.saveParameters(args.save_path+"/model{:09d}.model".format(it));

            print(time.strftime("%Y-%m-%d %H:%M:%S"), "TLOSS {:.5f}".format(loss));
            scorefile.write("IT {:d}, TLOSS {:.5f}\n".format(it, loss));

            scorefile.flush()
        
    ## Core training script(ee488b data)
    else:
        for it in range(1, args.max_epoch+1):

            clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

            print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training epoch {:d} with LR {:.5f} ".format(it,max(clr)));

            loss = trainer.train_network(trainLoader);

            if it % args.test_interval == 0:
            
                sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
                result = tuneThresholdfromScore(sc, lab, [1, 0.1]);

                print("IT {:d}, Val EER {:.5f}".format(it, result[1]));
                scorefile.write("IT {:d}, Val EER {:.5f}\n".format(it, result[1]));

                trainer.saveParameters(args.save_path+"/model{:09d}.model".format(it));

            print(time.strftime("%Y-%m-%d %H:%M:%S"), "TLOSS {:.5f}".format(loss));
            scorefile.write("IT {:d}, TLOSS {:.5f}\n".format(it, loss));

            scorefile.flush()

    scorefile.close();


# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    os.environ["CUDA_VISIBLE_DEVICES"]='{}'.format(args.gpu)
            
    if not(os.path.exists(args.save_path)):
        os.makedirs(args.save_path)

    main_worker(args)


if __name__ == '__main__':
    main()