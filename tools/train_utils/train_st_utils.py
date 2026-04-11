import torch
import os
import glob
import tqdm
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils
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

    # 1. 初始化 MM-MOS 控制器
    if tta_cfg and tta_cfg.ENABLED:
        logger.info('='*20 + ' MM-MOS START ' + '='*20)
        logger.info(f'理论复现：检测到 TTA 配置，方法: {tta_cfg.METHOD}')
        logger.info(f'多模态融合系数 Alpha: {tta_cfg.get("ALPHA", 0.4)}')
        # 实例化我们在 pcdet/tta_methods/mos.py 中定义的 MOS 类
        mos_worker = MOS(model, tta_cfg, logger, dataset=train_loader.dataset)
        # 将当前 run 的 ckpt_dir 显式传给 MOS，避免 _find_ckpt_dir 误命中历史目录
        if ckpt_save_dir is not None:
            mos_worker.run_ckpt_dir = str(ckpt_save_dir)
            logger.info(f'MM-MOS: 使用当前 run ckpt 目录进行 aggregation -> {mos_worker.run_ckpt_dir}')
    else:
        logger.error("未检测到有效的 TTA 配置，请检查 YAML 文件中的 TTA 字段")
        return

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

                # 步进学习率
                cur_scheduler.step(accumulated_iter)
                cur_lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0.0

                if optimizer is not None:
                    optimizer.zero_grad()
                
                # --- MM-MOS 核心逻辑调用 ---
                # optimize 内部处理了：
                # 1. 图像/点云特征提取 (BEVFusion Forward)
                # 2. 伪标签生成与 Hungarian 匹配
                # 3. 多模态余弦相似度计算 (Cos-Sim)
                # 4. 损失计算与反向传播 (Loss Backward)
                loss, tb_dict, disp_dict = mos_worker.optimize(batch_dict)
                if optimizer is not None:
                    clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                    optimizer.step()
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
                    ckpt_name = ckpt_save_dir / ('checkpoint_iter_%d' % samples_seen)
                    state = checkpoint_state(model, optimizer, cur_epoch, accumulated_iter)
                    save_checkpoint(state, filename=ckpt_name)
                    logger.info(f'MM-MOS: 已保存在线聚合 ckpt -> {ckpt_name}.pth')

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

    logger.info('='*20 + ' MM-MOS TTA COMPLETED ' + '='*20)
    return accumulated_iter