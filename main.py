import argparse
from ultralytics import YOLO


def main(model_name: str, image_path: str, output_path: str, method: str, port: int, server: str):
    model = YOLO(f"{method}://{server}:{port}/{model_name}", task="detect")

    results = model(image_path)

    results[0].save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some images with a model.')
    parser.add_argument('model_name', type=str, help='Name of the model to use')
    parser.add_argument('--image_path', type=str, required=False, default='man-holding-mug.jpg', help='Name of the image to process')
    parser.add_argument('--output_path', type=str, default='man-holding-mug-output.jpg', help='Path to save the output image')
    parser.add_argument('--method', type=str, choices=['grpc', 'http'], default='grpc', help="Method to use: 'grpc' or 'http'")
    parser.add_argument('--port', type=int, choices=[8001, 8000], default=8001, help="Port to use: 8001 or 8000")
    parser.add_argument('--server', type=str, default="localhost", help="Server address")

    args = parser.parse_args()

    main(args.model_name, args.image_path, args.output_path, args.method, args.port, args.server)
