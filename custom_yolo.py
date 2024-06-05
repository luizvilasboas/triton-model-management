from ultralytics import YOLO


class CustomYOLO(YOLO):
    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super().__init__(model, task, verbose)

        if task == 'pose':
            var_names = {0: "person"}
            self.predictor = self._smart_load("predictor")(_callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
            self.predictor.model.__dict__.update({"names": var_names, "kpt_shape": [17, 3]})
