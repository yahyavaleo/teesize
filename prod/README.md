# Production

This directory contains the optimized C++ code for model inference. The PyTorch model is converted to TorchScript by tracing and then loaded into the C++ program using the `libtorch` library. It also exports the model in ONNX format to ensure interoperability with frameworks like TensorRT.

## Usage Instructions

1. Download and extract the `libtorch` library.
2. Modify the `CMAKE_PREFIX_PATH` in `CMakeLists.txt` to specify the path of the extracted `libtorch` library.
3. Create a directory named `weights/` and place the pretrained model weights in there.
4. Run `python model.py` to generate the serialized TorchScript model and export the model in ONNX format.
5. Creata a directory named `build/`, navigate into it, and run `cmake ..`.
6. Navigate back to the root `prod/` directory and run `cmake --build build/ --config Release`.
7. Run the executable with the image file provided as argument, for example: `./teesize image.png`
