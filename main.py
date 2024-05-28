import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import argparse


class TritonClient:
    def __init__(self, url: str = "localhost:8001"):
        self.url = url
        self.client = self._get_triton_client()

    def _get_triton_client(self) -> grpcclient.InferenceServerClient:
        triton_client = grpcclient.InferenceServerClient(url=self.url, verbose=False)

        return triton_client

    def get_model_metadata(self, model_name: str):
        return self.client.get_model_metadata(model_name)

    def run_inference(self, model_name: str, input_image: np.ndarray):
        inputs = [grpcclient.InferInput("images", input_image.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_image)

        outputs = [
            grpcclient.InferRequestedOutput("num_detections"),
            grpcclient.InferRequestedOutput("detection_boxes"),
            grpcclient.InferRequestedOutput("detection_scores"),
            grpcclient.InferRequestedOutput("detection_classes")
        ]

        results = self.client.infer(
            model_name=model_name, inputs=inputs, outputs=outputs)
        return results


class ImageProcessor:
    @staticmethod
    def read_image(image_path: str, expected_image_shape) -> np.ndarray:
        expected_width, expected_height = expected_image_shape
        expected_length = min(expected_height, expected_width)
        original_image = cv2.imread(image_path)
        height, width, _ = original_image.shape
        length = max(height, width)
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = original_image
        scale = length / expected_length

        input_image = cv2.resize(image, (expected_width, expected_height))
        input_image = (input_image / 255.0).astype(np.float32)
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)

        return original_image, input_image, scale

    @staticmethod
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = f"({class_id}: {confidence:.2f})"
        color = (255, 0, 0)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


class InferencePipeline:
    def __init__(self, triton_client: TritonClient, image_processor: ImageProcessor):
        self.triton_client = triton_client
        self.image_processor = image_processor

    def process(self, image_path: str, model_name: str, image_output_path: str):
        expected_image_shape = self.triton_client.get_model_metadata(model_name).inputs[0].shape[-2:]
        original_image, input_image, scale = self.image_processor.read_image(image_path, expected_image_shape)
        results = self.triton_client.run_inference(model_name, input_image)

        num_detections = int(results.as_numpy("num_detections")[0])
        detection_boxes = results.as_numpy("detection_boxes")
        detection_scores = results.as_numpy("detection_scores")
        detection_classes = results.as_numpy("detection_classes")

        for index in range(num_detections):
            box = detection_boxes[index]
            self.image_processor.draw_bounding_box(
                original_image,
                detection_classes[index],
                detection_scores[index],
                round(box[0] * scale),
                round(box[1] * scale),
                round((box[0] + box[2]) * scale),
                round((box[1] + box[3]) * scale)
            )

        cv2.imwrite(image_output_path, original_image)


def main(image_path: str, model_name: str, url: str, image_output_path: str):
    triton_client = TritonClient(url)
    image_processor = ImageProcessor()
    pipeline = InferencePipeline(triton_client, image_processor)
    pipeline.process(image_path, model_name, image_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", "-i", type=str, default="man-holding-mug.jpg", help="Image input path")
    parser.add_argument("--image_output_path", "-o", type=str, default="man-holding-mug-output.jpg", help="Image output path")
    parser.add_argument("--model_name", "-m", type=str, default="yolov8n_ensemble")
    parser.add_argument("--url", type=str, default="localhost:8001")
    args = parser.parse_args()
    main(args.image_path, args.model_name, args.url, args.image_output_path)
