import os
import math
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from collections import defaultdict

from datasets import BehaviorSetSequentialRecDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

main_args_list = [
    'dataset',
    'config_files',
    'output_dir'
]

train_args_list = [
    'lr',
    'batch_size',
    'max_epochs',
    'log_freq',
    'eval_freq',
    'seed',
    'tensorboard_on',
    'run_dir'
]

model_args_list = [
    'init_std',
    'hidden_dims',
    'maxlen',
    'no',
    'dropout',

    ## SASRec
    'sals',
    'sal_heads',
    'sal_dropout',

    ## L-MSAB
    'lmsals',
    'lmsal_heads',
    'lmsal_dropout',
    'alpha',

    ## CBAF
    'cba_dropout',

    ## PBS-TPE
    'dcba_dropout',

    ## FFN
    'ffn_acti'
]

optimizer_args_list = [
    'weigth_decay',
    'adam_beta1',
    'adam_beta2'
]

scheduler_args_list = [
    'decay_factor',
    'min_lr',
    'patience'
]

loss_args_list = [
    'tau'
]

def clear_dict(d):
    if d is None:
        return None
    elif isinstance(d, list):
        return list(filter(lambda x: x is not None, map(clear_dict, d)))
    elif not isinstance(d, dict):
        return d
    else:
        r = dict(
                filter(lambda x: x[1] is not None,
                    map(lambda x: (x[0], clear_dict(x[1])),
                        d.items())))
        if not bool(r):
            return None
        return r

def setup_global_seed(SEED):
    print(f'Global SEED is setup as {SEED}.')

    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

def load_config_files(file_path, args):
    file_path = file_path + f'{args.batch_size}' + os.sep + f'{args.dataset}.yaml'
    # file_path = file_path + f'{args.dataset}.yaml'
    with open(file_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    yaml_config.update(clear_dict(vars(args)))
    return yaml_config

def show_args_info(args):
    pad = 26
    category_list = ['Main', 'Train', 'Model', 'Optimizer', 'Scheduler', 'Loss']
    whole_args_list = dict()
    whole_args_list['Main'] = main_args_list
    whole_args_list['Train'] = train_args_list
    whole_args_list['Model'] = model_args_list
    whole_args_list['Optimizer'] = optimizer_args_list
    whole_args_list['Scheduler'] = scheduler_args_list
    whole_args_list['Loss'] = loss_args_list

    args_info = set_color("*" * pad + f" Configure Info: " + f"*" * (pad+1) + '\n', 'red')
    for category in category_list:
        args_info += set_color(category + ' Hyper Parameters:\n', 'pink')
        args_info += '\n'.join([(set_color("{:<32}", 'cyan') + ' : ' +
                                 set_color("{:>35}", 'yellow')).format(arg_name, arg_value)
                                for arg_name, arg_value in vars(args).items()
                                if arg_name in whole_args_list[category]])
        args_info += '\n'

    print(args_info)

def check_output_path(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"{output_dir} has been automatically created..." )

def set_color(log, color, highlight=True):
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = '\033['
    if highlight:
        prev_log += '1;3'
    else:
        prev_log += '0;3'
    prev_log += str(index) + 'm'
    return prev_log + log + '\033[0m'

def determine_behavior_type(l):
    idx = len(l)
    while(idx):
        if int(l[idx-1]) == 1:
            return idx
        idx -= 1
    return idx

def data_partition(dataset_name):
    userset = set()
    itemset = set()
    all_behavior_type = 0

    user_train = defaultdict(list)
    user_valid = defaultdict(list)
    user_bvt = defaultdict(list)
    user_test = defaultdict(list)

    base_path = './data/' + dataset_name + os.sep

    if dataset_name[:3] == 'QKV' or dataset_name[:3] == 'QKA':

        train_data_path = base_path + 'train.txt'
        valid_data_path = base_path + 'valid.txt'
        test_data_path = base_path + 'test.txt'
        bvt_data_path = base_path + 'between_val_test.txt'

        path_list = [train_data_path, valid_data_path, bvt_data_path, test_data_path]
        dict_list = [user_train, user_valid, user_bvt, user_test]

        for path, dict in zip(path_list, dict_list):
            f = open(path, 'r')
            for line in f:
                u = line.rstrip().split(',')[0]
                i = line.rstrip().split(',')[1]
                u = int(u)
                i = int(i)

                behavior_set = [int(b) for b in line.rstrip().split(',')[2:]]
                bt = determine_behavior_type(behavior_set)

                # 这里我们忽略掉全为0的set(真实的负反馈)
                if bt != 0:
                    userset.add(u)
                    itemset.add(i)
                    all_behavior_type = max(bt, all_behavior_type)
                    dict[u].append([i, behavior_set])
                else:
                    continue
            f.close()

    print_dataset_msg = '\n'
    print_dataset_msg += set_color('Basic Statistics of ' +
                                   dataset_name + ' Dataset:\n', 'white')
    print_dataset_msg += set_color('# User Num: ' + str(len(userset)) + '\n', 'white')
    print_dataset_msg += set_color('# Max UserId: ' + str(max(userset)) + '\n', 'white')
    print_dataset_msg += set_color('# Item Num: ' + str(len(itemset)) + '\n', 'white')
    print_dataset_msg += set_color('# Max ItemId: ' + str(max(itemset)) + '\n', 'white')
    print_dataset_msg += set_color('# Behavior Type Num: ' + str(all_behavior_type) + '\n', 'white')

    total_seq_length = 0.0
    for u in user_train:
        total_seq_length += len(user_train[u])

    print_dataset_msg += set_color(f'# Avg Seq Length: {total_seq_length / len(user_train):.2f}' + '\n', 'white')
    print(print_dataset_msg)

    return (user_train, user_valid, user_bvt, user_test,
            userset, itemset, all_behavior_type)

def data_preparation(config_dict, data):
    config_dict['user_num'] = max(data[-3])
    config_dict['item_num'] = max(data[-2])
    config_dict['behavior_types'] = data[-1]

    train_dataset = BehaviorSetSequentialRecDataset(config_dict, data)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=config_dict['batch_size'])

    valid_dataset = BehaviorSetSequentialRecDataset(config_dict, data, data_type="valid")
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=config_dict['batch_size'])

    test_dataset = BehaviorSetSequentialRecDataset(config_dict, data, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=config_dict['batch_size'])

    return train_dataloader, valid_dataloader, test_dataloader, config_dict

def ndcg_k(pred, ground_truth, topk):
    ndcg = 0
    count = float(len(ground_truth))
    for uid in range(len(ground_truth)):
        if ground_truth[uid] == [0]:
            count -= 1.0
            continue
        k = min(topk, len(ground_truth[uid]))
        idcg = idcg_k(k)
        dcg_k = sum([int(pred[uid][j] in set(ground_truth[uid]))
                     / math.log(j+2, 2) for j in range(topk)])
        ndcg += dcg_k / idcg
    return ndcg / count

def idcg_k(k):
    '''
    Calculates the Ideal Discounted Cumulative Gain at k
    '''
    idcg = sum([1.0 / math.log(i+2, 2) for i in range(k)])
    if not idcg:
        return 1.0
    else:
        return idcg

def hr_k(pred, ground_truth, topk):
    hr = 0.0
    count = float(len(ground_truth))
    for uid in range(len(ground_truth)):
        if ground_truth[uid] == [0]:
            count -= 1.0
            continue
        pred_set = set(pred[uid][:topk])
        ground_truth_set = set(ground_truth[uid])
        hr += len(pred_set & ground_truth_set) / \
                  float(len(ground_truth_set))
    return hr / count

class EarlyStopping:
    '''
    Early stops the training if the test metrics doesn't improve after a given patience.
    '''
    def __init__(self, patience=20, verbose=True, delta=1e-5, save_path="./output"):
        '''
        Args:
        :param patience: How long to wait after last time Rec metrics improved.
                         Default: 20
        :param verbose: If True, prints a message for each metric improvement.
                         Default: True
        :param delta: Minimum change in the monitored quantity to qualify as improvement.
                      Default: 1e-5
        :param save_path: the folder path to save the best performance model.
        '''
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_scores = None
        self.best_valid_epoch = 0
        self.ndcg_10_max = 0
        self.hr_10_max = 0
        self.early_stop = False
        self.delta = delta
        self.save_path = save_path

    def __call__(self, scores, epoch, model):
        '''
        scores: [HR_5, NDCG_5, HR_10, NDCG_10, HR_20, NDCG_20]
        Here we focus on HR_10 & NDCG_10
        '''
        hr_10 = scores[2]
        ndcg_10 = scores[3]
        if self.best_scores is None:
            self.best_scores = scores
            self.best_valid_epoch = epoch
            self.save_checkpoint(scores, model)
        elif (ndcg_10 > self.ndcg_10_max + self.delta) and (hr_10 > self.hr_10_max + self.delta):
            self.save_checkpoint(scores, model)
            self.best_scores = scores
            self.best_valid_epoch = epoch
            self.counter = 0
        elif (ndcg_10 > self.ndcg_10_max + self.delta) or (hr_10 > self.hr_10_max + self.delta):
            self.counter = 0
            if ndcg_10 > self.ndcg_10_max + self.delta:
                print(f'NDCG@10 metrics obtained in this epoch exceeds from ({self.ndcg_10_max:.4f} --> {ndcg_10:.4f}), '
                      f'reset the counter...')
            else:
                print(f'HR@10 metrics obtained in this epoch exceeds from ({self.hr_10_max:.4f} --> {hr_10:.4f}), '
                      f'reset the counter...')
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


    def save_checkpoint(self, scores, model):
        '''
        Saves model when both NDCG and HR increases
        '''
        hr_10 = scores[2]
        ndcg_10 = scores[3]
        if self.verbose:
            print(f'NDCG@10 metrics has increased from ({self.ndcg_10_max:.4f} --> {ndcg_10:.4f})')
            print(f'HR@10 metrics has increased from ({self.hr_10_max:.4f} --> {hr_10:.4f}). Saving model ...')

        torch.save(model.state_dict(), self.save_path)
        self.ndcg_10_max = ndcg_10
        self.hr_10_max = hr_10