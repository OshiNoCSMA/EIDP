import yaml

dataset = 'QKA'
config_setting = {
    # main args:
    "dataset": 'QKA',
    "config_files": './config/',
    "output_dir": './output',

    # train args:
    "lr": 0.001,
    "batch_size": 1280,
    "max_epochs": 1000,
    "log_freq": 1,
    "eval_freq": 5,
    "seed": 2023,
    "tensorboard_on": False,
    "run_dir": './run',

    # optimizer args:
    "weight_decay": 0.0,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,

    # scheduler args:
    "decay_factor": 0.9,
    "min_lr": 0.000001,
    "patience": 5,

    # model args:
    "init_std": 0.02,
    "hidden_dims": 64,
    "maxlen": 50,
    "no": 1,
    'dropout': 0.4,

    ## SASRec model:
    "sals": 1,
    "sal_heads": 2,
    "sal_dropout": 0.7,

    ## L-MSAB module:
    "lmsals": 2,
    "lmsal_heads": 1,
    "lmsal_dropout": 0.5,
    "alpha": 8,

    ## CBAF module:
    "cba_dropout": 0.3,

    ## PBS-TPE module:
    "dcba_dropout": 0.8,

    ## FFN module:
    "ffn_acti": "relu",

    # Loss args:
    "tau": 1.0,
}

with open(f'./{dataset}.yaml', 'w', encoding='utf-8') as f:
   yaml.dump(data=config_setting, stream=f, allow_unicode=True)