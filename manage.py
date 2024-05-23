import argparse
import requests
import json

BASE_URL = "http://localhost:8000/v2/repository"

def load_models(models):
    for model in models:
        url = f"{BASE_URL}/models/{model}/load"
        
        print(f"Loading model: {model}")
        
        response = requests.post(url)

        if response.status_code == 200:
            print(f"Successfully loaded model: {model}")
        else:
            print(f"Failed to load model: {model}. Status code: {response.status_code}")


def unload_models(models):
    for model in models:
        url = f"{BASE_URL}/models/{model}/unload"

        print(f"Unloading model: {model}")

        response = requests.post(url)

        if response.status_code == 200:
            print(f"Successfully unloaded model: {model}")
        else:
            print(f"Failed to unload model: {model}. Status code: {response.status_code}")


def list_models():
    url = f"{BASE_URL}/index"

    response = requests.post(url)

    if response.status_code == 200:
        models = response.json()

        print(json.dumps(models, indent=4))

    else:
        print(f"Failed to list models. Status code: {response.status_code}")


def main():
    parser = argparse.ArgumentParser(description='Manage models')
    subparsers = parser.add_subparsers(dest='command', required=True)

    load_parser = subparsers.add_parser('load', help='Load models')
    load_parser.add_argument('models', nargs='+', help='Names of models to load')

    unload_parser = subparsers.add_parser('unload', help='Unload models')
    unload_parser.add_argument('models', nargs='+', help='Names of models to unload')

    subparsers.add_parser('list', help='List all models')

    args = parser.parse_args()

    if args.command == 'load':
        load_models(args.models)
    elif args.command == 'unload':
        unload_models(args.models)
    elif args.command == 'list':
        list_models()


if __name__ == "__main__":
    main()
