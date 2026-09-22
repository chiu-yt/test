import torch
import os
import glob
import tqdm
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.figure6_stream import Figure6StreamCapture
from pcdet.utils.efficiency_profiler import (
    EfficiencyProfileConfigurationError,
    EfficiencyProfileRun,
    EfficiencyProfiler,
)
from pcdet.tta_methods.codemerge import CodeMergeTTA
from pcdet.tta_methods.mos import MOS  # 导入你修改的 v31.0 MOS

def checkpoint_state(model, optimizer, epoch, it):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if hasattr(model, 'module'):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}


def save_checkpoint(state, filename='checkpoint'):
    if False: # 可以在这里添加分布式同步逻辑，单卡忽略
        pass
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def train_model_st(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                   start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir,
                   train_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                   max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, 
                   logger=None, tta_cfg=None, **kwargs):
    """
    MM-MOS Test-Time Adaptation 训练入口
    """
    accumulated_iter = start_iter
    figure6_capture = None

    # 1. 初始化 MM-MOS 控制器
    if tta_cfg and tta_cfg.ENABLED:
        logger.info('='*20 + ' MM-MOS START ' + '='*20)
        logger.info(f'理论复现：检测到 TTA 配置，方法: {tta_cfg.METHOD}')
        logger.info(f'多模态融合系数 Alpha: {tta_cfg.get("ALPHA", 0.4)}')
        # 实例化我们在 pcdet/tta_methods/mos.py 中定义的 MOS 类
        tta_method = str(tta_cfg.get('METHOD', 'mos')).lower()
        worker_cls = CodeMergeTTA if tta_method == 'codemerge' else MOS
        worker_kwargs = {}
        if tta_cfg.get('FIGURE6_CAPTURE', {}).get('ENABLED', False):
            figure6_capture = Figure6StreamCapture(
                kwargs['cfg'], kwargs.get('capture_provenance', {}), (ckpt_save_dir, rank))
            worker_kwargs['figure6_collector'] = figure6_capture.collector
        mos_worker = worker_cls(model, tta_cfg, logger, dataset=train_loader.dataset, **worker_kwargs)
        # 将当前 run 的 ckpt_dir 显式传给 MOS，避免 _find_ckpt_dir 误命中历史目录
        if ckpt_save_dir is not None:
            mos_worker.run_ckpt_dir = str(ckpt_save_dir)
            logger.info(f'MM-MOS: 使用当前 run ckpt 目录进行 aggregation -> {mos_worker.run_ckpt_dir}')
    else:
        logger.error("未检测到有效的 TTA 配置，请检查 YAML 文件中的 TTA 字段")
        return

    profile_cfg = tta_cfg.get('EFFICIENCY_PROFILE', {})
    profile_enabled = bool(profile_cfg.get('ENABLED', False))
    if profile_enabled and (
        rank != 0
        or (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
    ):
        raise EfficiencyProfileConfigurationError(
            'Train-based TTA efficiency profiling requires single-process execution'
        )
    run_cfg = kwargs['cfg']
    config_path = 'cfgs/%s/%s.yaml' % (run_cfg.EXP_GROUP_PATH, run_cfg.TAG)
    profiler = EfficiencyProfiler(
        profile_cfg,
        EfficiencyProfileRun(
            method='codemerge' if tta_method == 'codemerge' else 'refuse_tta',
            entrypoint='train.py',
            config=config_path,
            boundary=(
                'synchronized implementation online adaptation including '
                'method diagnostics and iteration checkpoint-bank writes; '
                'excludes dataloader iteration, load_data_to_gpu, scheduler, '
                'outer progress/TensorBoard logging, and epoch archival checkpoint'
            ),
            output_dir=ckpt_save_dir.parent,
            model_parameters=model.parameters(),
            optimizer=optimizer,
        ),
        logger=logger,
    )

    # 与 MOS-main 对齐：按 samples_seen 触发 iter checkpoint
    # SAVE_CKPT 为显式保存点，SAVE_CKPT_INTERVAL 为周期保存点
    save_ckpt_points = set()
    if tta_cfg is not None:
        raw_save_points = getattr(tta_cfg, 'SAVE_CKPT', [])
        try:
            save_ckpt_points = set(int(x) for x in list(raw_save_points))
        except Exception:
            save_ckpt_points = set()
    save_interval_iter = int(getattr(tta_cfg, 'SAVE_CKPT_INTERVAL', 0)) if tta_cfg is not None else 0

    # 2. 开始 Epoch 循环 (TTA 通常为 1 epoch)
    with tqdm.trange(start_epoch, total_epochs, desc='Epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # 学习率调整策略
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # 进入训练模式
            model.train()
            
            # 3. 核心迭代循环
            if rank == 0:
                pbar = tqdm.tqdm(total=total_it_each_epoch, leave=True, desc='TTA-MOS', dynamic_ncols=True)

            for it, batch_dict in enumerate(train_loader):
                # 与 MOS-main 保持一致：samples_seen = cur_it * batch_size（当前 iter 处理前）
                cur_batch_size = int(batch_dict.get('batch_size', getattr(train_loader, 'batch_size', 1)))
                samples_seen = int(it) * max(cur_batch_size, 1)
                batch_dict['samples_seen'] = samples_seen
                if figure6_capture is not None:
                    figure6_capture.begin(
                        batch_dict, (cur_epoch, accumulated_iter, samples_seen, cur_batch_size))

                # 步进学习率
                cur_scheduler.step(accumulated_iter)
                cur_lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0.0

                if profile_enabled:
                    load_data_to_gpu(batch_dict)
                profiler.begin_batch(cur_batch_size)
                profiler.begin_segment()

                if optimizer is not None:
                    optimizer.zero_grad()
                
                # --- MM-MOS 核心逻辑调用 ---
                # optimize 内部处理了：
                # 1. 图像/点云特征提取 (BEVFusion Forward)
                # 2. 伪标签生成与 Hungarian 匹配
                # 3. 多模态余弦相似度计算 (Cos-Sim)
                # 4. 损失计算与反向传播 (Loss Backward)
                if profile_enabled:
                    loss, tb_dict, disp_dict = mos_worker.optimize(
                        batch_dict, data_already_on_gpu=True
                    )
                else:
                    loss, tb_dict, disp_dict = mos_worker.optimize(batch_dict)
                if figure6_capture is not None:
                    figure6_capture.finish()
                if optimizer is not None:
                    clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                    optimizer.step()
                profiler.end_segment()
                # --------------------------

                accumulated_iter += 1

                # 与 MOS-main 参考实现对齐：按 samples_seen 保存 checkpoint_iter_*
                should_save_iter_ckpt = False
                if samples_seen in save_ckpt_points:
                    should_save_iter_ckpt = True
                if save_interval_iter > 0 and (samples_seen % save_interval_iter == 0):
                    should_save_iter_ckpt = True

                if (
                    rank == 0
                    and ckpt_save_dir is not None
                    and should_save_iter_ckpt
                ):
                    profiler.begin_segment()
                    ckpt_name = ckpt_save_dir / ('checkpoint_iter_%d' % samples_seen)
                    state = checkpoint_state(model, optimizer, cur_epoch, accumulated_iter)
                    save_checkpoint(state, filename=ckpt_name)
                    profiler.end_segment()
                    logger.info(f'MM-MOS: 已保存在线聚合 ckpt -> {ckpt_name}.pth')

                profiler.end_batch()

                # 更新进度条展示信息
                if rank == 0:
                    pbar.update()
                    # 在控制台显示 loss/lr + 当前相似度融合状态
                    log_dict = {
                        'loss': f'{float(loss):.4f}',
                        'lr': f'{cur_lr:.2e}'
                    }
                    if isinstance(disp_dict, dict):
                        log_dict.update(disp_dict)

                    pbar.set_postfix(log_dict)
                    tbar.set_postfix(log_dict)

                    # 记录 Tensorboard
                    if tb_log is not None:
                        tb_log.add_scalar('train/loss', loss, accumulated_iter)
                        tb_log.add_scalar('train/lr', cur_lr, accumulated_iter)
                        for key, val in tb_dict.items():
                            tb_log.add_scalar('train/' + key, val, accumulated_iter)
            
            if rank == 0:
                pbar.close()

            # 4. 保存模型权重 (TTA 结束后的快照)
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if len(ckpt_list) >= max_ckpt_save_num:
                    for i in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[i])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter)
                save_checkpoint(state, filename=ckpt_name)
                logger.info(f'MM-MOS: 已保存自适应后的模型权重 -> {ckpt_name}.pth')

    profiler.finalize()
    logger.info('='*20 + ' MM-MOS TTA COMPLETED ' + '='*20)
    return accumulated_iter
