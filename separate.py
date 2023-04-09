import onnx
import onnxruntime
model_path = "model_f16.onnx"
model = onnx.load(model_path)

# replace problematic characters
for initializer in model.graph.initializer:
    if ":" in initializer.name:
        new_name = initializer.name.replace(":", "_")
        initializer.name = new_name



# Save the modified model to a new ONNX file
output_path = "separated//model.onnx"
onnx.save(model, output_path,save_as_external_data=True, all_tensors_to_one_file=False)