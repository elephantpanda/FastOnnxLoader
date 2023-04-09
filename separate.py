import onnx
import onnxruntime
model_path = "model_f16.onnx"
model = onnx.load(model_path)

# replace problematic characters
for node in model.graph.node:
    if ':' in node.name:
        node.name = node.name.replace(':', '_')
    for i, input_name in enumerate(node.input):
        if ':' in input_name:
            node.input[i] = input_name.replace(':', '_')
    for i, output_name in enumerate(node.output):
        if ':' in output_name:
            node.output[i] = output_name.replace(':', '_')

# Loop through all of the initializers in the model and rename any initializers with colons in their names
for init in model.graph.initializer:
    if ':' in init.name:
        init.name = init.name.replace(':', '_')



# Save the modified model to a new ONNX file (must be in a differnt folder!)
output_path = "separated//model.onnx"
onnx.save(model, output_path,save_as_external_data=True, all_tensors_to_one_file=False)
