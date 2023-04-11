#Created with help of CHAT GPT
import onnx
import json
# Load the ONNX model
model = onnx.load("separated_quant//model.onnx")

# Create an empty dictionary to store the scale and zero point values
values_dict = {}

# Iterate through all the tensors in the model
for tensor in model.graph.initializer:
    # Check if the tensor ends with "_scale" or "_zero_point"
    if tensor.name.endswith("_scale"):
        # Get the value and key names from the tensor
        value_name = tensor.name.replace("_scale", "")
        # Get the value from the tensor
        data = tensor.float_data or tensor.int32_data or tensor.int64_data
        # Add the value to the dictionary
        if value_name not in values_dict:
            values_dict[value_name] = {}
        values_dict[value_name]["scale"] = data[0]
    elif tensor.name.endswith("_zero_point"):
        # Get the value and key names from the tensor
        value_name = tensor.name.replace("_zero_point", "")
        # Get the value from the tensor
        data = tensor.float_data or tensor.int32_data or tensor.int64_data
        # Add the value to the dictionary
        if value_name not in values_dict:
            values_dict[value_name] = {}
        values_dict[value_name]["zero_point"] = data[0]

# Print the values dictionary
print(values_dict)


file_name = "scale_zeros.json"

# Open the file for writing
with open(file_name, "w") as f:
    # Write the dictionary to the file as a JSON string
    json.dump(values_dict, f)