from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.exporters.onnx import validate_model_outputs
from optimum.exporters import TasksManager
from optimum.exporters.onnx.model_configs import BertOnnxConfig, DistilBertOnnxConfig
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)

import os
from pathlib import Path
import torch
from torch import nn, Tensor
import onnx
import onnxruntime


# MODEL_PATH=onnx-model/google-bert-chinese/onnx-artifacts \
#     MODEL_FILE=training_model.onnx \
#     AUTO_TOKENIZER_MODEL_NAME_OR_PATH=onnx-model/google-bert-chinese/base_model \
#     python onnx-model/google-bert-chinese/hfoptimum-check.py

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
output = os.environ.get("MODEL_PATH", f"{script_dir}/base_model")
model_file = os.environ.get("MODEL_FILE", "model.onnx")
auto_tokenizer_model_name_or_path = os.environ.get(
    "AUTO_TOKENIZER_MODEL_NAME_OR_PATH", output
)

# optimized_model_path = "google-bert/bert-base-chinese"
optimized_model_path = Path(output)
optimized_model = optimized_model_path / model_file
print(f"Will load: {optimized_model}")

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

tokenizer = AutoTokenizer.from_pretrained(auto_tokenizer_model_name_or_path)
model = ORTModelForSequenceClassification.from_pretrained(optimized_model_path)
inputs = tokenizer(
    "5月8日付款成功，当当网显示5月10日发货，可是至今还没看到货物，也没收到任何通知，简不知怎么说好！！！",
    return_tensors="pt",
)
outputs = model(**inputs)
logits = outputs.logits
print(list(logits.shape))

probs = nn.Softmax(logits)
print(f"{probs}")
