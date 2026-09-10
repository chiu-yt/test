class TransFusionLogitCapture:
    def __init__(self, model):
        self.model = model.module if hasattr(model, 'module') else model
        self.logits = None
        self._original_predict = None

    def __enter__(self):
        dense_head = getattr(self.model, 'dense_head', None)
        if dense_head is None or not hasattr(dense_head, 'predict'):
            raise NotImplementedError('Tent requires a dense_head.predict method that returns heatmap logits')
        original_predict = dense_head.predict
        self._original_predict = original_predict

        def wrapped_predict(inputs):
            result = original_predict(inputs)
            if 'heatmap' not in result:
                raise KeyError('Tent could not find pre-NMS classification logits in dense_head.predict result["heatmap"]')
            self.logits = result['heatmap']
            return result

        dense_head.predict = wrapped_predict
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        dense_head = getattr(self.model, 'dense_head', None)
        if dense_head is not None and self._original_predict is not None:
            dense_head.predict = self._original_predict
        return False
