from ultralytics import YOLO
import argparse
import shutil


def convert_to_onnx(model_path: str, output_path: str):
    model = YOLO(model_path)

    onnx_file = model.export(format='onnx', opset=15)

    if onnx_file:
        if output_path:
            shutil.move(onnx_file, output_path)
            print(f"> Arquivo ONNX movido para {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert from .pt to .onnx")
    parser.add_argument("model", help="Path of .pt model")
    parser.add_argument("--output", "-o", help="Output path of .onnx model")

    args = parser.parse_args()

    model = args.model
    output = args.output

    print(f"> Convertendo {model} para ONNX")
    convert_to_onnx(model, output)
    print(f"> Convertido {model} para ONNX")


if __name__ == '__main__':
    main()
