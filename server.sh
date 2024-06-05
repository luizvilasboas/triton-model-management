#!/usr/bin/env bash

docker run -it --rm --net=host -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ./models:/models -v ./labels:/mnt --name triton-server triton-inference-server-manager
