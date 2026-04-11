import _init_path
import argparse
import datetime
import glob
import os
import types
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from experiment_doc_utils import add_experiment_doc_args, maybe_update_experiment_docs, summarize_best_ckpt
from train_utils.optimization import build_optimizer, build_scheduler

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    add_experiment_doc_args(parser)

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def _force_module_eval_and_freeze(module):
    for param in module.parameters():
        param.requires_grad = False

    module.eval()

    def _train_keep_eval(self, mode=True):
        nn.Module.train(self, False)
        return self

    module.train = types.MethodType(_train_keep_eval, module)


def apply_tta_freeze_strategy(model, cfg, logger):
    tta_cfg = cfg.get('TTA', None)
    if tta_cfg is None:
        return []

    freeze_cfg = tta_cfg.get('FREEZE', None)
    if freeze_cfg is None or not freeze_cfg.get('ENABLED', False):
        return []

    model_ref = model.module if hasattr(model, 'module') else model
    module_names = list(freeze_cfg.get('MODULES', []))
    frozen_modules = []

    logger.info('TTA Freeze: 开始执行冻结策略')
    for module_name in module_names:
        if not hasattr(model_ref, module_name):
            logger.warning(f'TTA Freeze: 模型中不存在模块 `{module_name}`，已跳过')
            continue

        cur_module = getattr(model_ref, module_name)
        if cur_module is None:
            logger.warning(f'TTA Freeze: 模块 `{module_name}` 为 None，已跳过')
            continue

        param_num = sum(p.numel() for p in cur_module.parameters())
        _force_module_eval_and_freeze(cur_module)
        frozen_modules.append(module_name)
        logger.info(f'TTA Freeze: 已冻结 `{module_name}`，参数量={param_num}')

    total_params = sum(p.numel() for p in model_ref.parameters())
    trainable_params = sum(p.numel() for p in model_ref.parameters() if p.requires_grad)
    logger.info(
        f'TTA Freeze: 冻结模块={frozen_modules} | 可训练参数={trainable_params}/{total_params} '
        f'({trainable_params / max(total_params, 1):.2%})'
    )
    return frozen_modules
def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')
    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # 创建 Dataloader
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    # 创建模型
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # 创建优化器
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # 加载权重
    start_epoch = it = 0
    last_epoch = -1
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
        
    if cfg.get('TTA', None) and cfg.TTA.ENABLED:
        logger.info("TTA 模式：重置 start_epoch 为 0 以执行自适应")
        start_epoch = 0
        it = 0
        last_epoch = 0

        frozen_modules = apply_tta_freeze_strategy(model, cfg, logger)
        if len(frozen_modules) > 0:
            optimizer = build_optimizer(model, cfg.OPTIMIZATION)
            logger.info('TTA Freeze: 已基于冻结后的可训练参数重新构建优化器')

    model.train()
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    # 创建学习率调度器
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # --- 核心修改部分：准备训练参数 ---
    train_kwargs = {
        'model': model,
        'optimizer': optimizer,
        'train_loader': train_loader,
        'model_func': model_fn_decorator(),
        'lr_scheduler': lr_scheduler,
        'optim_cfg': cfg.OPTIMIZATION,
        'start_epoch': start_epoch,
        'total_epochs': args.epochs,
        'start_iter': it,
        'rank': cfg.LOCAL_RANK,
        'tb_log': tb_log,
        'ckpt_save_dir': ckpt_dir,
        'train_sampler': train_sampler,
        'lr_warmup_scheduler': lr_warmup_scheduler,
        'ckpt_save_interval': args.ckpt_save_interval,
        'max_ckpt_save_num': args.max_ckpt_save_num,
        'merge_all_iters_to_one_epoch': args.merge_all_iters_to_one_epoch, 
        'logger': logger, 
        'logger_iter_interval': args.logger_iter_interval,
        'ckpt_save_time_interval': args.ckpt_save_time_interval,
        'use_logger_to_record': not args.use_tqdm_to_record, 
        'show_gpu_stat': not args.wo_gpu_stat,
        'use_amp': args.use_amp,
        'cfg': cfg
    }

    # --- 根据 TTA 开关选择不同的训练入口 ---
    if cfg.get('TTA', None) and cfg.TTA.ENABLED:
        from train_utils.train_st_utils import train_model_st
        logger.info('理论复现：已成功挂载 TTA 自适应训练流程 (MM-MOS)')
        train_model_st(tta_cfg=cfg.TTA, **train_kwargs)
    else:
        from train_utils.train_utils import train_model
        logger.info('执行标准有监督训练流程')
        train_model(**train_kwargs)

    # --- 训练结束后的清理与评估 ---
    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************Start evaluation**********************')
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    eval_ckpt_type = 'iter' if (cfg.get('TTA', None) and cfg.TTA.ENABLED) else 'epoch'
    args.eval_ckpt_type = eval_ckpt_type
    logger.info(f'Auto Eval: 当前将按 `{eval_ckpt_type}` 级别 checkpoint 自动评估')
    
    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    best_summary = summarize_best_ckpt(eval_output_dir, logger, ckpt_type=eval_ckpt_type)
    maybe_update_experiment_docs(
        args=args,
        cfg=cfg,
        eval_output_dir=eval_output_dir,
        logger=logger,
        output_dir=output_dir,
        best_summary=best_summary,
        source='tools/train.py'
    )

if __name__ == '__main__':
    main()
