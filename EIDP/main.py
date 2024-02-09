# Formal Version

import argparse

from utils import set_color, load_config_files, show_args_info, setup_global_seed, \
    check_output_path, data_partition, data_preparation
from model import *
from trainer import EIDPTrainer

def str2bool(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def main():
    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument('--dataset', default='QKV', type=str)
    parser.add_argument('--config_files', type=str, default='./config/', help='config yaml files')
    parser.add_argument("--do_eval", action="store_true")

    # model args
    parser.add_argument("--no", type=int, default=1, help="batch number of the training terminal")
    parser.add_argument("--dropout", type=float, help="hidden dropout p")

    ## SASRec
    parser.add_argument("--sals", type=int, help="num of self attention layers")
    parser.add_argument("--sal_heads", type=int, help="num of attention heads of SAL")
    parser.add_argument("--sal_dropout", type=float, help="hidden dropout p of SASRec")

    ## L-MSAB
    parser.add_argument("--lmsals", type=int, help="num of Light Multi-head Self Attention Layers")
    parser.add_argument("--lmsal_heads", type=int, help="num of attention heads of L-MSAL")
    parser.add_argument("--lmsal_dropout", type=float, help="hidden dropout p of L-MSAL")
    parser.add_argument("--alpha", type=int, help="sample num for Query in ProbSparse Attention")

    ## CBAF
    parser.add_argument("--cba_dropout", type=float, help="hidden dropout p of CBA mechanism")

    ## PBS-TPE
    parser.add_argument("--dcba_dropout", type=float, help="hidden dropout p of DCBA attention")

    ## FFN
    parser.add_argument("--ffn_acti", type=str, help="activation function of FFN")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of optimizer")
    parser.add_argument("--batch_size", type=int, default=1280, help="batch size of training phase")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--tensorboard_on", type=str2bool, default=False,
                       help="whether to launch the tensorboard and record the scalar to analyse the training phase")

    # loss function args
    # parser.add_argument("--tau", type=float, default=1.0, help="temperature coefficient in InfoNCE loss function")

    args = parser.parse_args()
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print(set_color('Using CUDA: ' + str(args.cuda_condition) + '\n', 'green'))

    config_dict = load_config_files(args.config_files, args)
    show_args_info(argparse.Namespace(**config_dict))

    setup_global_seed(config_dict['seed'])
    check_output_path(config_dict['output_dir'])

    data = data_partition(config_dict['dataset'])
    train_dataloader, valid_dataloader, test_dataloader, config_dict = data_preparation(config_dict, data)

    # model = SASRec(config=config_dict)
    model = EIDP(config=config_dict)
    trainer = EIDPTrainer(model, train_dataloader, valid_dataloader, test_dataloader, config_dict)

    if config_dict['do_eval']:
        trainer.load()
        _, test_info = trainer.test(record=False)
        print(set_color(f'\nFinal Test Metrics: ' +
                        test_info + '\n', 'pink'))
    else:
        trainer.train()


if __name__ == '__main__':
    main()