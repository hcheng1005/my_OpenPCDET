import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model



# # log to file
# logger.info('**********************Start logging**********************')
# gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
# logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)


#加载配置文件,文件定义了数据路径、类型个数、训练参数等等详细内容
cfg_file = "./cfgs/kitti_models/pointpillar.yaml"
cfg_from_yaml_file(cfg_file, cfg)
# print(cfg)

dist_train = False 
BatchSize = 1
Workers = 4
# -----------------------create dataloader & network & optimizer---------------------------
train_set, train_loader, train_sampler = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=BatchSize,
    dist=dist_train, workers=Workers,
    logger=None,
    training=True,
    merge_all_iters_to_one_epoch=False,
    total_epochs=None
)

model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
# if args.sync_bn:
#     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.cuda()

# print(model)

optimizer = build_optimizer(model, cfg.OPTIMIZATION)

# load checkpoint if it is possible
ckpt_dir = "./tempckpt/"
start_epoch = it = 0
last_epoch = -1

lr_scheduler, lr_warmup_scheduler = build_scheduler(
    optimizer, total_iters_each_epoch=len(train_loader), total_epochs=50,
    last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
)

model.train() 

train_model(
    model,
    optimizer,
    train_loader,
    model_func=model_fn_decorator(),
    lr_scheduler=lr_scheduler,
    optim_cfg=cfg.OPTIMIZATION,
    start_epoch=0,
    total_epochs=50,
    start_iter=it,
    rank=0,
    tb_log=None,
    ckpt_save_dir=ckpt_dir,
    train_sampler=train_sampler,
    lr_warmup_scheduler=lr_warmup_scheduler,
    ckpt_save_interval=1,
    max_ckpt_save_num=50,
    merge_all_iters_to_one_epoch=False
)