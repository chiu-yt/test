import json
from datetime import datetime
from pathlib import Path


EXPERIMENT_RECORD_BEGIN = '<!-- AUTO-EXPERIMENT-LOG:BEGIN -->'
EXPERIMENT_RECORD_END = '<!-- AUTO-EXPERIMENT-LOG:END -->'
AGENTS_PROMOTED_BEGIN = '<!-- AUTO-AGENTS-PROMOTED:BEGIN -->'
AGENTS_PROMOTED_END = '<!-- AUTO-AGENTS-PROMOTED:END -->'
REGISTRY_PATH = 'experiment_registry.json'


def add_experiment_doc_args(parser):
    parser.add_argument(
        '--disable_doc_autoupdate', action='store_true',
        help='disable automatic updates to Experiment Record.md and AGENTS.md'
    )
    parser.add_argument(
        '--exp_name', type=str, default=None,
        help='human-readable experiment name used in auto-updated docs'
    )
    parser.add_argument(
        '--exp_note', type=str, default='',
        help='short note that describes what changed in this experiment'
    )
    parser.add_argument(
        '--promote_to_agents', action='store_true',
        help='promote this experiment into the auto-maintained AGENTS.md summary'
    )


def _safe_float(val):
    try:
        return float(val)
    except Exception:
        return None


def _extract_ckpt_id_from_path(path_str, ckpt_type='epoch'):
    import re

    if ckpt_type == 'iter':
        patterns = [r'checkpoint_iter_(\d+)', r'iter_(\d+)']
    else:
        patterns = [r'checkpoint_epoch_(\d+)', r'epoch_(\d+)']

    for pattern in patterns:
        match = re.search(pattern, path_str)
        if match:
            return int(match.group(1))
    return None


def _infer_ckpt_type_and_id(path_str):
    iter_id = _extract_ckpt_id_from_path(path_str, ckpt_type='iter')
    if iter_id is not None:
        return 'iter', iter_id

    epoch_id = _extract_ckpt_id_from_path(path_str, ckpt_type='epoch')
    if epoch_id is not None:
        return 'epoch', epoch_id

    return None, None


def _clean_text(text):
    if text is None:
        return ''
    return ' '.join(str(text).split())


def _format_metric(name, value):
    if value is None:
        return '%s=NA' % name
    return '%s=%.4f' % (name, value)


def _collect_scalar_metrics(result_dict):
    if not isinstance(result_dict, dict):
        return {}

    metrics = {}
    nds = _safe_float(result_dict.get('NDS', result_dict.get('nd_score', result_dict.get('nds', None))))
    map_val = _safe_float(result_dict.get('mAP', result_dict.get('mean_ap', result_dict.get('map', None))))

    if nds is not None:
        metrics['NDS'] = nds
    if map_val is not None:
        metrics['mAP'] = map_val

    if metrics:
        return metrics

    for key, val in result_dict.items():
        if key.startswith('diag/') or key.startswith('recall/'):
            continue
        scalar_val = _safe_float(val)
        if scalar_val is None:
            continue
        metrics[key] = scalar_val
        if len(metrics) >= 4:
            break

    return metrics


def summarize_best_ckpt(eval_output_dir, logger, ckpt_type='epoch'):
    metric_files = sorted(Path(eval_output_dir).rglob('metrics_summary.json'))
    if len(metric_files) == 0:
        if logger is not None:
            logger.warning('Auto Eval Summary: 在 %s 下未找到 metrics_summary.json' % eval_output_dir)
        return None

    rows = []
    for metric_file in metric_files:
        try:
            data = json.loads(metric_file.read_text(encoding='utf-8'))
        except Exception:
            continue

        nds = _safe_float(data.get('nd_score', data.get('NDS', data.get('nds', None))))
        map_val = _safe_float(data.get('mean_ap', data.get('mAP', data.get('map', None))))
        if nds is None and map_val is None:
            continue

        ckpt_id = _extract_ckpt_id_from_path(str(metric_file), ckpt_type=ckpt_type)
        rows.append({
            'ckpt_id': ckpt_id,
            'NDS': nds,
            'mAP': map_val,
            'metrics_file': str(metric_file)
        })

    if len(rows) == 0:
        if logger is not None:
            logger.warning('Auto Eval Summary: 未解析到有效指标')
        return None

    rows.sort(key=lambda x: (
        x['NDS'] if x['NDS'] is not None else -1,
        x['mAP'] if x['mAP'] is not None else -1
    ), reverse=True)
    best = rows[0]

    if logger is not None:
        if best['ckpt_id'] is not None:
            logger.info(
                'Auto Eval Summary: Best %s=checkpoint_%s_%s | NDS=%s | mAP=%s'
                % (
                    ckpt_type, ckpt_type, best['ckpt_id'],
                    '%.4f' % best['NDS'] if best['NDS'] is not None else 'NA',
                    '%.4f' % best['mAP'] if best['mAP'] is not None else 'NA'
                )
            )
        else:
            logger.info(
                'Auto Eval Summary: Best %s (id unresolved) | NDS=%s | mAP=%s'
                % (
                    ckpt_type,
                    '%.4f' % best['NDS'] if best['NDS'] is not None else 'NA',
                    '%.4f' % best['mAP'] if best['mAP'] is not None else 'NA'
                )
            )

    summary = {
        'ckpt_type': ckpt_type,
        'best': best,
        'top_k': rows[:10]
    }
    summary_path = Path(eval_output_dir) / ('best_%s_summary.json' % ckpt_type)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    if logger is not None:
        logger.info('Auto Eval Summary: 已写入 %s' % summary_path)
    return summary


def _load_registry(registry_path):
    if not registry_path.exists():
        return []

    try:
        data = json.loads(registry_path.read_text(encoding='utf-8'))
    except Exception:
        return []

    return data if isinstance(data, list) else []


def _save_registry(registry_path, records):
    registry_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')


def _build_record(args, cfg, eval_output_dir, output_dir=None, best_summary=None, single_eval_metrics=None, source='tools/test.py'):
    eval_output_dir = Path(eval_output_dir)
    output_dir = Path(output_dir) if output_dir is not None else eval_output_dir.parent

    metrics = {}
    ckpt_type = None
    ckpt_id = None
    ckpt_label = None
    metrics_file = None
    record_mode = 'single_eval'

    if best_summary is not None:
        record_mode = 'best_summary'
        ckpt_type = best_summary.get('ckpt_type')
        best = best_summary.get('best', {})
        ckpt_id = best.get('ckpt_id')
        ckpt_label = 'checkpoint_%s_%s' % (ckpt_type, ckpt_id) if ckpt_id is not None else 'best_%s' % ckpt_type
        metrics = {
            'NDS': _safe_float(best.get('NDS')),
            'mAP': _safe_float(best.get('mAP'))
        }
        metrics = {k: v for k, v in metrics.items() if v is not None}
        metrics_file = best.get('metrics_file')
    else:
        metrics = _collect_scalar_metrics(single_eval_metrics)
        ckpt_path = getattr(args, 'ckpt', None)
        if ckpt_path:
            ckpt_type, ckpt_id = _infer_ckpt_type_and_id(ckpt_path)
            ckpt_label = Path(ckpt_path).name
        if metrics_file is None:
            metric_files = sorted(eval_output_dir.rglob('metrics_summary.json'))
            if metric_files:
                metrics_file = str(metric_files[0])

    if len(metrics) == 0:
        return None

    exp_name = args.exp_name if getattr(args, 'exp_name', None) else '%s/%s' % (cfg.TAG, getattr(args, 'extra_tag', 'default'))
    note = _clean_text(getattr(args, 'exp_note', ''))
    cfg_file = getattr(args, 'cfg_file', None)
    extra_tag = getattr(args, 'extra_tag', None)
    eval_tag = getattr(args, 'eval_tag', None)

    key_parts = [
        str(Path(cfg_file)) if cfg_file else 'unknown_cfg',
        str(eval_output_dir.resolve()),
        ckpt_label or 'no_ckpt_label',
        record_mode
    ]
    record_key = '||'.join(key_parts)

    return {
        'record_key': record_key,
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'source': source,
        'cfg_file': cfg_file,
        'exp_name': exp_name,
        'extra_tag': extra_tag,
        'eval_tag': eval_tag,
        'note': note,
        'promote_to_agents': bool(getattr(args, 'promote_to_agents', False)),
        'record_mode': record_mode,
        'eval_all': bool(getattr(args, 'eval_all', False)),
        'output_dir': str(output_dir),
        'eval_output_dir': str(eval_output_dir),
        'ckpt_type': ckpt_type,
        'ckpt_id': ckpt_id,
        'ckpt_label': ckpt_label,
        'metrics': metrics,
        'metrics_file': metrics_file
    }


def _upsert_record(records, record):
    replaced = False
    new_records = []
    for item in records:
        if item.get('record_key') == record['record_key']:
            new_records.append(record)
            replaced = True
        else:
            new_records.append(item)

    if not replaced:
        new_records.append(record)

    new_records.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return new_records


def _replace_auto_section(doc_path, begin_marker, end_marker, body, title):
    doc_path = Path(doc_path)
    text = doc_path.read_text(encoding='utf-8') if doc_path.exists() else ''

    if begin_marker in text and end_marker in text:
        before, rest = text.split(begin_marker, 1)
        _, after = rest.split(end_marker, 1)
        new_text = before + begin_marker + '\n' + body.rstrip() + '\n' + end_marker + after
    else:
        block = '\n\n%s\n%s\n%s\n%s\n' % (title, begin_marker, body.rstrip(), end_marker)
        new_text = text.rstrip() + block

    doc_path.write_text(new_text, encoding='utf-8')


def _render_experiment_record_body(records, limit=12):
    if len(records) == 0:
        return '- No auto-recorded experiment runs yet. Run `tools/train.py` or `tools/test.py` to populate this section.'

    lines = []
    for record in records[:limit]:
        metrics = record.get('metrics', {})
        metric_chunks = []
        for name in ['NDS', 'mAP']:
            if name in metrics:
                metric_chunks.append(_format_metric(name, metrics[name]))
        if not metric_chunks:
            metric_chunks = [_format_metric(k, v) for k, v in list(metrics.items())[:4]]

        ckpt_type = record.get('ckpt_type')
        ckpt_id = record.get('ckpt_id')
        if ckpt_type is not None and ckpt_id is not None:
            ckpt_desc = 'best_%s=%s' % (ckpt_type, ckpt_id)
        else:
            ckpt_desc = record.get('ckpt_label', 'single_eval')

        promoted = 'yes' if record.get('promote_to_agents') else 'no'
        lines.append(
            '- `%s` `%s` `%s` `%s` `%s` `%s`'
            % (
                record.get('updated_at', 'unknown_time').replace('T', ' '),
                record.get('source', 'unknown_source'),
                record.get('cfg_file', 'unknown_cfg'),
                record.get('extra_tag', 'default'),
                ckpt_desc,
                ' '.join(metric_chunks + ['promoted=%s' % promoted])
            )
        )
        lines.append('  - exp: `%s`' % record.get('exp_name', 'unnamed'))
        if record.get('note'):
            lines.append('  - note: %s' % record['note'])
        lines.append('  - eval_dir: `%s`' % record.get('eval_output_dir', 'unknown_eval_dir'))

    return '\n'.join(lines)


def _record_score(record):
    metrics = record.get('metrics', {})
    nds = _safe_float(metrics.get('NDS'))
    map_val = _safe_float(metrics.get('mAP'))
    return (nds if nds is not None else -1, map_val if map_val is not None else -1)


def _render_agents_body(records):
    promoted_records = [record for record in records if record.get('promote_to_agents')]
    if len(promoted_records) == 0:
        return '- No promoted experiment summaries yet. Use `--promote_to_agents --exp_note "why this run matters"` when a result is stable enough to guide future agents.'

    promoted_records.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    best_record = max(promoted_records, key=_record_score)

    def _one_line(record, prefix):
        metrics = record.get('metrics', {})
        metric_chunks = []
        for name in ['NDS', 'mAP']:
            if name in metrics:
                metric_chunks.append(_format_metric(name, metrics[name]))
        if not metric_chunks:
            metric_chunks = [_format_metric(k, v) for k, v in list(metrics.items())[:4]]

        ckpt_type = record.get('ckpt_type')
        ckpt_id = record.get('ckpt_id')
        if ckpt_type is not None and ckpt_id is not None:
            ckpt_desc = 'best_%s=%s' % (ckpt_type, ckpt_id)
        else:
            ckpt_desc = record.get('ckpt_label', 'single_eval')

        line = '- %s `%s` `%s` `%s` `%s` `%s`' % (
            prefix,
            record.get('updated_at', 'unknown_time').replace('T', ' '),
            record.get('cfg_file', 'unknown_cfg'),
            record.get('extra_tag', 'default'),
            ckpt_desc,
            ' '.join(metric_chunks)
        )
        if record.get('note'):
            line += '; note: %s' % record['note']
        return line

    lines = [_one_line(best_record, 'Current promoted best:')]
    for record in promoted_records[:3]:
        if record.get('record_key') == best_record.get('record_key'):
            continue
        lines.append(_one_line(record, 'Recent promoted run:'))

    return '\n'.join(lines)


def maybe_update_experiment_docs(args, cfg, eval_output_dir, logger=None, output_dir=None,
                                 best_summary=None, single_eval_metrics=None, source='tools/test.py'):
    if getattr(args, 'disable_doc_autoupdate', False):
        if logger is not None:
            logger.info('Experiment Doc Update: disabled by --disable_doc_autoupdate')
        return None

    if getattr(cfg, 'LOCAL_RANK', 0) != 0:
        return None

    record = _build_record(
        args=args,
        cfg=cfg,
        eval_output_dir=eval_output_dir,
        output_dir=output_dir,
        best_summary=best_summary,
        single_eval_metrics=single_eval_metrics,
        source=source
    )
    if record is None:
        if logger is not None:
            logger.warning('Experiment Doc Update: no usable scalar metrics were found, skip markdown updates')
        return None

    root_dir = Path(cfg.ROOT_DIR)
    registry_path = root_dir / REGISTRY_PATH
    records = _load_registry(registry_path)
    records = _upsert_record(records, record)
    _save_registry(registry_path, records)

    _replace_auto_section(
        root_dir / 'Experiment Record.md',
        EXPERIMENT_RECORD_BEGIN,
        EXPERIMENT_RECORD_END,
        _render_experiment_record_body(records),
        '## 0. Auto-maintained Experiment Ledger'
    )
    _replace_auto_section(
        root_dir / 'AGENTS.md',
        AGENTS_PROMOTED_BEGIN,
        AGENTS_PROMOTED_END,
        _render_agents_body(records),
        '## Auto-Promoted Findings'
    )

    if logger is not None:
        logger.info('Experiment Doc Update: synced %s and %s' % (root_dir / 'Experiment Record.md', root_dir / 'AGENTS.md'))
    return record
