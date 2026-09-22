"""Route training entrypoints and attach auditable TTA CLI provenance."""

from pcdet.utils.figure6_provenance import build_capture_provenance

def launch_training(cfg, args, train_kwargs):
    logger = train_kwargs['logger']
    if cfg.get('TTA', None) and cfg.TTA.ENABLED:
        from .train_st_utils import train_model_st
        logger.info('理论复现：已成功挂载 TTA 自适应训练流程 (MM-MOS)')
        train_model_st(
            tta_cfg=cfg.TTA,
            capture_provenance=build_capture_provenance(
                args.cfg_file, args.ckpt, args.fix_random_seed),
            **train_kwargs
        )
        return
    from .train_utils import train_model
    logger.info('执行标准有监督训练流程')
    train_model(**train_kwargs)
