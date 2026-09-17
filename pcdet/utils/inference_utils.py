_ANNOTATION_KEYS = {
    'gt', 'gt_boxes', 'gt_names', 'ground_truth', 'annotations', 'annos',
    'sample_annotation_tokens',
}


def build_inference_batch(batch_dict):
    return {
        key: value for key, value in batch_dict.items()
        if key.lower() not in _ANNOTATION_KEYS
        and not key.lower().startswith('gt_')
        and 'ground_truth' not in key.lower()
    }


def forward_without_annotations(model, batch_dict):
    inference_batch = build_inference_batch(batch_dict)
    output = model(inference_batch)
    batch_dict.update(inference_batch)
    return output
