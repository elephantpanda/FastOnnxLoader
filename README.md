# FastOnnxLoader 🚀👩‍🚀
Loads in onnx files with less RAM. (Not necessarily faster)

This script is a test to load an ONNX file in OnnxRuntime with less RAM by first loading in an ONNX with no embedded weights, then sequentially loading in the weight files one by one.

Workflow
===
1. export the torch model using `torch.onnx.export()` which gives you your basic model.onnx file
2. If the file does not have separate weights, then use `python separate.py` on model.onnx
3. Use `python clear.py` on this new model.onnx file
4. Use the C# code to load in the `model_without_weights.onnx` and sequentially load in the weight files.

separate.py
===
This takes an onnx file and separates it into separate weight files. It makes sure to rename weights with illegal characters such as ":"

clear.py
===
If you already have an onnx file with separated weights (for example it might be a >2GB onnx). Then you can just use clear.py to create a **model_without_weights.onnx** file.


Issues
===
1. Appears to be a memory leak in BindInput that doesn't clear up the system RAM after pushing weights to the GPU.
2. Current bottleneck is reading in a tensor of Float16s.


