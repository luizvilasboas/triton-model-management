FROM nvcr.io/nvidia/tritonserver:23.02-py3

RUN pip install opencv-python && apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

CMD ["tritonserver", "--model-repository=/models"]
