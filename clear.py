#saves a model without the weights

import onnx

# Load the ONNX model
model = onnx.load("model.onnx")

# Create a list of inputs to replace the initializers
inputs = []

# Iterate over the initializers and create new inputs
for initializer in model.graph.initializer:
    input = onnx.helper.make_tensor_value_info(
        name=initializer.name,
        elem_type=initializer.data_type,
        shape=initializer.dims,
    )
    inputs.append(input)

# Add the new inputs to the model graph
model.graph.input.extend(inputs)

# Remove the initializers from the model graph
model.graph.ClearField('initializer')

# Save the modified ONNX model
onnx.save(model, "model_without_weights.onnx")