# FastOnnxLoader
Loads in onnx files with less RAM.

This script is a test to load an ONNX file in OnnxRuntime with less RAM by first loading in an ONNX with no embedded weights, then sequentially loading in the weight files one by one.
