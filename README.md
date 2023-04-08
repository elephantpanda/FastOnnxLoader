# FastOnnxLoader 🚀👩‍🚀
Loads in onnx files with less RAM.

This script is a test to load an ONNX file in OnnxRuntime with less RAM by first loading in an ONNX with no embedded weights, then sequentially loading in the weight files one by one.

Steps
===

1. export the torch model with `torch.onnx.export(..)` and the flag `export_params=True` [This creates an onnx file with embeded weights **"model.onnx"**)
2. export the torch model with `torch.onnx.export(..)` and the flag `export_params=False` [This creates a small onnx file without weights **"model_no_weights.onnx"**)

3. load the **model.onnx** and then `onnx.save(model, output_path, save_as_external_data=True, all_tensors_to_one_file=False) `[This takes the large **model.onnx** file and separates all the weights into separate files **"model_separated.onnx"** plus lots of weight files]

4. Create a session using the **"model_no_weights.onnx"**
5. Iterate through all the 'inputs' (each weight becomes an an input), get the name of the input, use this to load the weights file from the disk. Use IOBinding to bind these weights to that input. 
7. Release the RAM from the IOBinding (I haven't worked out how to do this yet!)
8. Bind the actual input data
9. Bind the output
10. Run the inference with RunWithBindingAndNames()


Issues
===
1. BindInput(...) seems to not release RAM after it pushed the data onto the GPU
2. loading in the weight-less onnx sometimes takes longer than expected


Improvements
===
I am considering using:
```
using (FixedBufferOnnxValue value = FixedBufferOnnxValue.CreateFromTensor(new DenseTensor<ushort>(new Memory<ushort>(shorts,0,dimSize), dims)))
{
    binding.BindInput(key, value);
}
```
Although I'm not sure if this will create different values if the shorts array is the same.
