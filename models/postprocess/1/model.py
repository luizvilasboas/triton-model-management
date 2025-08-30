import numpy as np
import json
import triton_python_backend_utils as pb_utils
import cv2


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded."""
        self.model_config = json.loads(args['model_config'])
        self.input_tensor_names = [input_config["name"] for input_config in self.model_config["input"]]

        self.num_detections_dtype = self._get_output_dtype("num_detections")
        self.detection_boxes_dtype = self._get_output_dtype("detection_boxes")
        self.detection_scores_dtype = self._get_output_dtype("detection_scores")
        self.detection_classes_dtype = self._get_output_dtype("detection_classes")

        self.score_threshold = 0.25
        self.nms_threshold = 0.45

        self.labels = self._get_labels()
    
    def _get_labels(self):
        models = self.input_tensor_names
        
        labels = {}

        for model in models:
            labels[model] = []

            with open(f"/mnt/{model}.txt", "r") as file:
                for line in file:
                    labels[model].append(line.rstrip('\n'))
        
        return labels

    def _get_output_dtype(self, output_name):
        output_config = pb_utils.get_output_config_by_name(self.model_config, output_name)

        return pb_utils.triton_string_to_numpy(output_config['data_type'])

    def execute(self, requests):
        """`execute` receives a list of pb_utils.InferenceRequest as the only argument."""
        responses = []
        for request in requests:
            input_tensors = self._get_input_tensors(request)
            detections = self._process_tensors(input_tensors)
            response = self._create_response(detections)
            responses.append(response)

        return responses

    def _get_input_tensors(self, request):
        input_tensors = {name: pb_utils.get_input_tensor_by_name(request, name).as_numpy() for name in self.input_tensor_names}
        return input_tensors

    def _process_tensors(self, input_tensors):
        all_detections = []

        for input_name, tensor in input_tensors.items():
            output = cv2.transpose(tensor[0])
            boxes, scores, class_names = [], [], []

            rows = output.shape[0]
            for i in range(rows):
                classes_scores = output[i][4:]
                _, max_score, _, (max_class_idx, _) = cv2.minMaxLoc(classes_scores)
                if max_score >= self.score_threshold:
                    box = [output[i][0] - (0.5 * output[i][2]), output[i][1] - (0.5 * output[i][3]), output[i][2], output[i][3]]
                    boxes.append(box)
                    scores.append(max_score)
                    class_names.append(self.labels[input_name][max_class_idx])

            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, scores, self.score_threshold, self.nms_threshold)
            else:
                indices = []

            detections = {'boxes': [], 'scores': [], 'class_names': [], 'num_detections': 0}
            if len(indices) > 0:
                for i in indices:
                    idx = i[0] if isinstance(i, np.ndarray) else i
                    detections['boxes'].append(boxes[idx])
                    detections['scores'].append(scores[idx])
                    detections['class_names'].append(class_names[idx])

                detections['num_detections'] = len(detections['boxes'])

            all_detections.append(detections)

        aggregated_detections = {'boxes': [], 'scores': [], 'class_names': [], 'num_detections': 0}
        for detection in all_detections:
            aggregated_detections['boxes'].extend(detection['boxes'])
            aggregated_detections['scores'].extend(detection['scores'])
            aggregated_detections['class_names'].extend(detection['class_names'])
            aggregated_detections['num_detections'] += detection['num_detections']

        return aggregated_detections

    def _create_response(self, detections):
        num_detections_tensor = pb_utils.Tensor("num_detections", np.array([detections['num_detections']], dtype=self.num_detections_dtype))
        detection_boxes_tensor = pb_utils.Tensor("detection_boxes", np.array(detections['boxes'], dtype=self.detection_boxes_dtype))
        detection_scores_tensor = pb_utils.Tensor("detection_scores", np.array(detections['scores'], dtype=self.detection_scores_dtype))
        detection_classes_tensor = pb_utils.Tensor("detection_classes", np.array(detections['class_names'], dtype=self.detection_classes_dtype))

        return pb_utils.InferenceResponse(output_tensors=[num_detections_tensor, detection_boxes_tensor, detection_scores_tensor, detection_classes_tensor])

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded."""
        pass
