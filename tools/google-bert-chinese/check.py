import onnxruntime
import sys

model = sys.argv[1]

# 加载模型
session = onnxruntime.InferenceSession(model)

# 获取模型输入的详细信息
input_info = session.get_inputs()

for input in input_info:
    name = input.name
    shape = input.shape
    dtype = input.type
    print(f"Input Name: {name}")
    print(f"Shape: {shape}")
    print(f"Data Type: {dtype}")
