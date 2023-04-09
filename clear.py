#saves a model without the weights

import onnx
import sys

# Load the ONNX model
model = onnx.load("separated//model.onnx")


# Create a list of inputs to replace the initializers
inputs = []

# Create a set of names of initializers that have been replaced with inputs
replaced_names = set()

# Iterate over the initializers and create new inputs if the initializer data is over 640 bytes
for initializer in model.graph.initializer:
    if sys.getsizeof(initializer.raw_data) > 640+34: #(For some reason weights of less than 640 bytes don't get externalized)
        #print(str(initializer.name)+"-->"+str(sys.getsizeof(initializer.raw_data) ))
        input = onnx.helper.make_tensor_value_info(
            name=initializer.name,
            elem_type=initializer.data_type,
            shape=initializer.dims,
        )
        inputs.append(input)
        replaced_names.add(initializer.name)
    else:
        print("Skipping "+str(initializer.name) +"-->"+str(sys.getsizeof(initializer.raw_data)))

# Add the new inputs to the model graph
model.graph.input.extend(inputs)

# Remove the initializers from the model graph, except for the ones that have been replaced with inputs
new_initializers = [initializer for initializer in model.graph.initializer if initializer.name not in replaced_names]
del model.graph.initializer[:]
model.graph.initializer.extend(new_initializers)

# Save the modified ONNX model
onnx.save(model, "separated//model_without_weights.onnx")
