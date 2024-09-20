from experiments.exp_ETT import Exp_ETT
import argparse
import os
import torch
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Argument parser
parser = argparse.ArgumentParser(description='Experiment on ETT dataset')

# -------  Dataset settings --------------
parser.add_argument('--model', type=str, default='DPAD_SE',
                    help='model of the experiment')
parser.add_argument('--data', type=str, required=False, default='weather', 
                    choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'traffic', 'weather', 'electricity'], 
                    help='name of dataset')
parser.add_argument('--root_path', type=str, default='./datasets/long/', 
                    help='root path of the data file')
parser.add_argument('--data_path', type=str, default='weather.csv', 
                    help='location of the data file')
parser.add_argument('--features', type=str, default='S', 
                    choices=['S', 'M'], help='features S is univariate, M is multivariate')
parser.add_argument('--target', type=str, default='OT', 
                    help='target feature')
parser.add_argument('--checkpoints', type=str, default='exp/run_ETT/', 
                    help='location of model checkpoints')

# -------  Model settings --------------
parser.add_argument('--seq_len', type=int, default=336,
                    help='look back window')
parser.add_argument('--label_len', type=int, default=0,
                    help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=336,
                    help='prediction sequence length, horizon')
parser.add_argument('--enc_hidden', type=int, default=336, 
                    help='hidden size of DRD module')
parser.add_argument('--dec_hidden', type=int, default=336, 
                    help='hidden size of IF module')
parser.add_argument('--dropout', type=float, default=0.5, 
                    help='dropout')
parser.add_argument('--levels', type=int, default=2)
parser.add_argument('--K_IMP', type=int, default=6)
parser.add_argument('--num_heads', type=int, default=1)

# -------  Training settings --------------
parser.add_argument('--cols', type=str, nargs='+', 
                    help='file list')
parser.add_argument('--num_workers', type=int, default=1, 
                    help='data loader num workers')
parser.add_argument('--itr', type=int, default=0, 
                    help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, 
                    help='train epochs')
parser.add_argument('--batch_size', type=int, default=512, 
                    help='batch size of train input data')
parser.add_argument('--patience', type=int, default=5, 
                    help='early stopping patience')
parser.add_argument('--lr', type=float, default=0.0001, 
                    help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='mae', 
                    help='loss function')
parser.add_argument('--lradj', type=int, default=1, 
                    help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', 
                    help='use automatic mixed precision training', default=False)
parser.add_argument('--save', type=bool, default=False, 
                    help='save the output results')
parser.add_argument('--model_name', type=str, default='DPAD_GCN')
parser.add_argument('--RIN', default=1, type=int, help='ReVIN')
parser.add_argument('--evaluate', type=int, default=0)

args = parser.parse_args()

def main():

    torch.manual_seed(79)  # reproducible
    torch.cuda.manual_seed_all(79)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    mae_ = []
    mse_ = []

    Exp = Exp_ETT

    if args.evaluate:
        setting = '{}_{}_ft{}_sl{}_pl{}_imp{}_lr{}_bs{}_eh{}_dh{}_l{}_dp{}_itr0_K{}'.format(
            args.model, args.data, args.features, args.seq_len, args.pred_len, args.K_IMP, args.lr, args.batch_size, 
            args.enc_hidden, args.dec_hidden, args.levels, args.dropout, args.K_IMP)
        
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        mse, mae = exp.test(setting, evaluate=1)
        print('Final mean normed mse:{:.4f},mae:{:.4f}'.format(mse, mae))
    
    else:
        if args.itr:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_ft{}_sl{}_pl{}_imp{}_lr{}_bs{}_eh{}_dh{}_l{}_dp{}_itr{}_K{}'.format(
                    args.model, args.data, args.features, args.seq_len,  args.pred_len, args.K_IMP, args.lr, 
                    args.batch_size, args.enc_hidden, args.dec_hidden, args.levels, args.dropout, ii, args.K_IMP)
                
                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                mae, mse = exp.test(setting)
                mae_.append(mae)
                mse_.append(mse)

                torch.cuda.empty_cache()

            print('Final mean normed mse:{:.4f}, std mse:{:.4f}, mae:{:.4f}, std mae:{:.4f}'.format(
                np.mean(mse_), np.std(mse_), np.mean(mae_), np.std(mae_)))
            print('Final min normed mse:{:.4f}, mae:{:.4f}'.format(
                min(mse_), min(mae_)))
        else:
            setting = '{}_{}_ft{}_sl{}_pl{}_imp{}_lr{}_bs{}_eh{}_dh{}_l{}_dp{}_itr0_K{}'.format(
                args.model, args.data, args.features, args.seq_len, args.pred_len, args.K_IMP, args.lr, args.batch_size, 
                args.enc_hidden, args.dec_hidden, args.levels, args.dropout, args.K_IMP)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mae, mse = exp.test(setting)
            print('Final mean normed mse:{:.4f},mae:{:.4f}'.format(mse, mae))

if __name__ == '__main__':
    main()
