import argparse
from custom_yolo import CustomYOLO as YOLO


def main(model_name: str, image_path: str, output_path: str, method: str, port: int, server: str, task: str):
    model = YOLO(f"{method}://{server}:{port}/{model_name}", task=task)

    if task == 'pose':
        results = model.track(image_path, persist=True)
    else:
        results = model(image_path)

    results[0].save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images with a model.')
    parser.add_argument('model_name', type=str, help='Name of the model to use')
    parser.add_argument('task', type=str, help='Task for model')
    parser.add_argument('--image_path', '-i', type=str, required=False, default='man-holding-mug.jpg', help='Name of the image to process')
    parser.add_argument('--output_path', '-o', type=str, default='man-holding-mug-output.jpg', help='Path to save the output image')
    parser.add_argument('--method', '-m', type=str, choices=['grpc', 'http'], default='grpc', help="Method to use: 'grpc' or 'http'")
    parser.add_argument('--port', '-p', type=int, choices=[8001, 8000], default=8001, help="Port to use: 8001 or 8000")
    parser.add_argument('--server', '-s', type=str, default="localhost", help="Server address")

    args = parser.parse_args()

    main(args.model_name, args.image_path, args.output_path, args.method, args.port, args.server, args.task)
