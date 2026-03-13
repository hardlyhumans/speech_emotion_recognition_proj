import torch
import argparse
from models import SER_Model, MLP
from data import Preprocessor, SER_Dataset
from torch.utils.data import dataloader

#Define train,test,val logic here
def train(args):
    pass

def val(args):
    pass

def test(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    # Add test mode ...
    train_parser = subparsers.add_parser('train')
    # Add more train args if needed...
    train_parser.add_argument('--epochs',type=int,default=20)
    train_parser.add_argument('--batch-size', type=int, default=64)
    train_parser.add_argument('--device', type=str, default="cpu")
    train_parser.add_argument('--checkpoint-path', type = str, default='./')
    train_parser.add_argument('--freeze-backbone', action="store_true", default=False)
    args = parser.parse_args()

    args.model = SER_Model() # model; mention backbone and head 
    
    if args.mode == 'train':
        args.train_dataset = SER_Dataset() # train dataset
        train(args)


