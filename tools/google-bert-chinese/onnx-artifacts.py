import math

import onnx

from onnxruntime.training import artifacts
import torch
from torch import nn, Tensor
from torch.nn import functional as F

import sys

onnx_model = onnx.load(sys.argv[1])
requires_grad = [param.name for param in onnx_model.graph.initializer]

artifacts.generate_artifacts(
	onnx_model,
	requires_grad=requires_grad,
	frozen_params=[],
	loss=artifacts.LossType.CrossEntropyLoss,
	optimizer=artifacts.OptimType.AdamW,
	artifact_directory=sys.argv[2]
)


