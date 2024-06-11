# triton-model-management

This project explores implementing model management in Triton Inference Server. Its objective is to allow continuous loading and unloading of models dynamically.

## Getting Started

1. Clone the Repository:

   Use Git to clone the repository locally:

   ```
   git clone https://gitlab.com/olooeez/triton-model-management.git
   ```

   > **Note:** Git Large File Storage (LFS) is required for this repository. Install it using `git lfs install` before cloning.

2. Install Dependencies:

   Navigate to the project directory and install necessary dependencies using pip:

   ```
   cd triton-model-management
   pip install -r requirements.txt
   ```

3. Build the Docker Image:

   Construct a Docker image to encapsulate the project environment:

   ```
   docker build -t triton-inference-server-manager .
   ```

4. Create a Container:

   Execute the `server.sh` script to set up the Triton Inference Server within a container:

   ```
   bash server.sh
   ```

5. Compile Protocol Buffers file

   Run the command below to compile a `config_pb2.py` for the protocol buffers:

   ```
   protoc --python_out=. config.proto
   ```

   > **Note**: You need to have `protoc` installed. Use `sudo apt install protobuf-compiler` to install it.

6. Run the Model Management:

   Run the `manage.py` script with the load or unload commands to load or unload models:

   ```
   python3 manage.py [load|unload] MODEL_NAME
   ```

   When using the `load` command, pass the config.pbtxt file. Here is an example:

   ```
   python3 manage.py load yolov8n --config configs/yolov8n.pbtxt
   ```

   To create a model ensemble dynamically use command `ensemble` like the one below:

   ```
   python3 manage.py ensemble MODEL_NAME...  # To load model ensemble
   python3 manage.py unensemble              # To unload model ensemble
   ```

   You can also use the `list` command to list all models and their status:

   ```
   python3 manage.py list
   ```

7. Run the Client:

   Run the `main.py` script to infer an image with a specific model:

   ```
   python3 main.py
   ```

## Contributing

If you're interested in contributing to this project, feel free to open a merge request. We welcome all forms of collaboration!

## License

This project is available under the [The Unlicense](https://gitlab.com/olooeez/triton-model-management/-/blob/main/LICENSE). For more information, please see the LICENSE file.
