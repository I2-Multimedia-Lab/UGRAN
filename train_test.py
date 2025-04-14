from train import train
from test import test
import os
import argparse

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--train', default=False, type=bool, help='Train or not')
    parser.add_argument('--data_root', default='./datasets/', type=str, help='data path')
    parser.add_argument('--train_epochs', default=60, type=int, help='total training epochs')
    parser.add_argument('--img_size', default=384, type=int, help='network input size')
    parser.add_argument('--method', default='UGRAN-R', type=str, help='M3Net with different backbone')
    parser.add_argument('--pretrained_model', default='./pretrained_model/', type=str, help='load Pretrained model')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    #parser.add_argument('--step1epochs', default=100, type=int, help='train epochs for the step 1')
    #parser.add_argument('--step2epochs', default=20, type=int, help='train epochs for the step 2')
    parser.add_argument('--trainset', default='DUTS-TR', type=str, help='Trainging set')
    parser.add_argument('--save_model', default='savepth/', type=str, help='save model path')

    # test
    parser.add_argument('--test', default=False, type=bool, help='Test or not')
    parser.add_argument('--save_test', default='preds/', type=str, help='save saliency maps path')
    parser.add_argument('--test_methods', type=str, default='DUTS-TE+DUT-O+ECSSD+HKU-IS+PASCAL-S+SOD')

    # eval
    # Need PySODMetrics
    parser.add_argument('--eval', default=False, type=bool, help='Test or not')
    parser.add_argument('--record', default='./record.txt', type=str, help='record file')
    args = parser.parse_args()

    if args.train:
        train(args=args)
    if args.test:
        test(args=args)
    if args.eval:
        os.system('python ../eval/eval.py \
              --method-json eval/preds.json \
              --dataset-json eval/datasets.json \
              --record-txt ' + args.record)
