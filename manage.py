import argparse
import requests
import json
from jsonify import pbtxt_to_dict, create_ensemble_config


BASE_URL = "http://localhost:8000/v2"


def load_model(model, config = None):
    if config:
        config = pbtxt_to_dict(config)

        config = json.dumps(config)

        payload = {
            "parameters": {
                "config": config
            }
        }

    url = f"{BASE_URL}/repository/models/{model}/load"

    print(f"Loading model: {model}")

    if config:
        response = requests.post(url, json=payload)
    else:
        response = requests.post(url)

    if response.status_code == 200:
        print(f"Successfully loaded model: {model}")
    else:
        print(f"Failed to load model: {model}. Status code: {response.status_code}")


def unload_model(model):
    url = f"{BASE_URL}/repository/models/{model}/unload"

    print(f"Unloading model: {model}")

    response = requests.post(url)

    if response.status_code == 200:
        print(f"Successfully unloaded model: {model}")
    else:
        print(f"Failed to unload model: {model}. Status code: {response.status_code}")


def list_models():
    url = f"{BASE_URL}/repository/index"

    response = requests.post(url)

    if response.status_code == 200:
        models = response.json()

        print(json.dumps(models, indent=4))
    else:
        print(f"Failed to list models. Status code: {response.status_code}")


def list_config_model(model):
    url = f"{BASE_URL}/models/{model}/config"

    response = requests.get(url)

    if response.status_code == 200:
        config = response.json()

        print(json.dumps(config, indent=4))
    else:
        print(f"Failed to list model config. Status code: {response.status_code}")


def load_model_ensemble(models):
    ensemble_json, postprocess_json = create_ensemble_config(models)

    configs = {
        "postprocess": postprocess_json,
        "yolov8n_ensemble": ensemble_json,
    }

    for model, config in configs.items():
        config = json.dumps(config)

        payload = {
            "parameters": {
                "config": config
            }
        }

        url = f"{BASE_URL}/repository/models/{model}/load"

        print(f"Loading model: {model}")

        if config:
            response = requests.post(url, json=payload)
        else:
            response = requests.post(url)

        if response.status_code == 200:
            print(f"Successfully loaded model: {model}")
        else:
            print(f"Failed to load model: {model}. Status code: {response.status_code}")


def unload_ensemble():
    unload_model("yolov8n_ensemble")
    unload_model("postprocess")


def main():
    parser = argparse.ArgumentParser(description='Manage models')
    subparsers = parser.add_subparsers(dest='command', required=True)

    load_parser = subparsers.add_parser('load', help='Load models')
    load_parser.add_argument('model', type=str, help='Name of model to load')
    load_parser.add_argument('--config', type=str, required=False, help='Config of model to load')

    unload_parser = subparsers.add_parser('unload', help='Unload models')
    unload_parser.add_argument('model', type=str, help='Name of model to unload')

    subparsers.add_parser('list', help='List all models')

    config_parser = subparsers.add_parser('config', help='List config')
    config_parser.add_argument('model', type=str, help='Name of model to list config')

    ensemble_parser = subparsers.add_parser('ensemble', help='Load model ensemble')
    ensemble_parser.add_argument('models', nargs='+', help='Model to load into ensemble')

    unload_ensemble_parser = subparsers.add_parser('unensemble', help='Unload model ensemble')

    args = parser.parse_args()

    if args.command == 'load':
        load_model(args.model, args.config)
    elif args.command == 'unload':
        unload_model(args.model)
    elif args.command == 'list':
        list_models()
    elif args.command == 'config':
        list_config_model(args.model)
    elif args.command == 'ensemble':
        load_model_ensemble(args.models)
    elif args.command == 'unensemble':
        unload_ensemble()


if __name__ == "__main__":
    main()
