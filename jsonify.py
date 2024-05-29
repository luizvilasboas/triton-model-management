from google.protobuf import text_format
from google.protobuf.json_format import MessageToJson
import config_pb2
import json


def pbtxt_to_dict(pbtxt_file):
    message = config_pb2.ModelConfig()

    with open(pbtxt_file, 'r') as f:
        pbtxt_content = f.read()
        text_format.Parse(pbtxt_content, message)

    json_content = MessageToJson(
        message, including_default_value_fields=True, preserving_proto_field_name=True)

    json_content = json.loads(json_content)

    if 'backend' in json_content and not json_content['backend']:
        del json_content['backend']
    if 'instance_group' in json_content and not json_content['instance_group']:
        del json_content['instance_group']

    return json_content


def create_ensemble_config(model_names: list[str]) -> tuple[dict, dict]:
    ensemble_config = {
        "name": "yolov8n_ensemble",
        "platform": "ensemble",
        "input": [
            {
                "name": "images",
                "dims": [1, 3, 640, 640],
                "data_type": "TYPE_FP32"
            }
        ],
        "output": [
            {
                "name": "num_detections",
                "data_type": "TYPE_INT32",
                "dims": [1]
            },
            {
                "name": "detection_boxes",
                "dims": [1000, 4],
                "data_type": "TYPE_FP32"
            },
            {
                "name": "detection_scores",
                "dims": [1000],
                "data_type": "TYPE_FP32"
            },
            {
                "name": "detection_classes",
                "data_type": "TYPE_INT32",
                "dims": [1000]
            }
        ],
        "ensemble_scheduling": {
            "step": []
        }
    }

    for idx, model_name in enumerate(model_names):
        step = {
            "model_name": model_name,
            "model_version": -1,
            "input_map": {
                "images": "images",
            },
            "output_map": {
                "output0": f"{model_name}_output"
            }
        }

        ensemble_config["ensemble_scheduling"]["step"].append(step)

    postprocess_step = {
        "model_name": "postprocess",
        "model_version": -1,
        "input_map": {},
        "output_map": {
            "detection_classes": "detection_classes",
            "num_detections": "num_detections",
            "detection_boxes": "detection_boxes",
            "detection_scores": "detection_scores"
        }
    }

    for idx, model_name in enumerate(model_names):
        postprocess_step["input_map"][f"INPUT_{idx}"] = f"{model_name}_output"

    ensemble_config["ensemble_scheduling"]["step"].append(postprocess_step)

    postprocess_inputs = []
    for idx in range(len(model_names)):
        input_entry = {
            "name": f"INPUT_{idx}",
            "dims": [-1, -1, -1],
            "data_type": "TYPE_FP32"
        }

        postprocess_inputs.append(input_entry)

    postprocess_config = {
        "name": "postprocess",
        "backend": "python",
        "input": postprocess_inputs,
        "output": [
            {
                "name": "num_detections",
                "data_type": "TYPE_INT32",
                "dims": [1]
            },
            {
                "name": "detection_boxes",
                "dims": [1000, 4],
                "data_type": "TYPE_FP32"
            },
            {
                "name": "detection_scores",
                "dims": [1000],
                "data_type": "TYPE_FP32"
            },
            {
                "name": "detection_classes",
                "data_type": "TYPE_INT32",
                "dims": [1000]
            }
        ],
        "instance_group": [
            {
                "count": 0,
                "kind": "KIND_CPU"
            }
        ]
    }

    return ensemble_config, postprocess_config


if __name__ == '__main__':
    model_names = ["yolov8n_mug", "yolov8n"]
    ensemble_json, postprocess_json = create_ensemble_config(model_names)
    # print(json.dumps(ensemble_json, indent=4))
    print(json.dumps(postprocess_json, indent=4))
