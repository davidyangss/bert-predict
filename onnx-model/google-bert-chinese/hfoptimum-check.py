
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import validate_model_outputs
from optimum.exporters import TasksManager
from optimum.exporters.onnx.model_configs import BertOnnxConfig, DistilBertOnnxConfig
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification

import os
from pathlib import Path
import torch
import onnx
import onnxruntime

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
output = os.environ.get("MODEL_PATH", f"{script_dir}/base_model")
model_file = os.environ.get("MODEL_FILE", "model.onnx")

# optimized_model_path = "google-bert/bert-base-chinese"
optimized_model_path = Path(output)
optimized_model = optimized_model_path / model_file

onnx_model = onnx.load(optimized_model)
onnx.checker.check_model(onnx_model)
print("Model is valid ONNX")

session = onnxruntime.InferenceSession(optimized_model)
input_info = session.get_inputs()
for input in input_info:
    name = input.name
    shape = input.shape
    dtype = input.type
    print(f"Input Name: {name}, Data Type: {dtype}, Shape: {shape}")
for output in session.get_outputs():
    name = output.name
    shape = output.shape
    dtype = output.type
    print(f"Output Name: {name}, Data Type: {dtype}, Shape: {shape}")

tokenizer = AutoTokenizer.from_pretrained(optimized_model_path)
model = ORTModelForSequenceClassification.from_pretrained(optimized_model_path)
inputs = tokenizer("涉及国家重要资源、经济利益等方面的档案、材料", return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
print(list(logits.shape))