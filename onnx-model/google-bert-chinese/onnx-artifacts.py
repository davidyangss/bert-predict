import math

import onnx

from onnxruntime.training import artifacts
import torch
from torch import nn, Tensor
from torch.nn import functional as F

import sys
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

log_level = "INFO"
logger.setLevel(log_level)

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
onnx_model = os.environ.get("ONNX_MODEL", f"{script_dir}/base_model/model.onnx")

output = os.environ.get("ONNX_OUTPUT", f"{script_dir}/onnx-artifacts")
output_path = Path(output)
output_path.mkdir(parents=True, exist_ok=True)

onnx_model = onnx.load(onnx_model)
for domain in onnx_model.opset_import:
	print(f"domain: {domain}")
	if domain.domain == "" or domain.domain == "ai.onnx":
		break
        
requires_grad = [param.name for param in onnx_model.graph.initializer]

artifacts.generate_artifacts(
	onnx_model,
	requires_grad=requires_grad,
	frozen_params=[],
	loss=artifacts.LossType.BCEWithLogitsLoss,
	optimizer=artifacts.OptimType.AdamW,
    ort_format=True,
    # additional_output_names=["logits"],
	artifact_directory=output
)

print("Done. generate artifacts")


