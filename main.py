from ultralytics import YOLO

model = YOLO("grpc://localhost:8001/yolov8n", task="detect")

results = model("man-holding-mug.jpg")

results[0].save("man-holding-mug-output.jpg")
