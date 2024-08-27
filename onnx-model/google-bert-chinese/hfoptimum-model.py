# https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/contribute#example-adding-support-for-bert

import sys
import os
from pathlib import Path
from typing import Dict

from optimum.exporters import TasksManager
from optimum.exporters.onnx import main_export, export, validate_model_outputs
from optimum.exporters.onnx.config import TextDecoderOnnxConfig, TextEncoderOnnxConfig
from optimum.exporters.onnx.model_configs import BertOnnxConfig, DistilBertOnnxConfig
from optimum.utils import NormalizedTextConfig, DummyPastKeyValuesGenerator, DummyTextInputGenerator
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)
from transformers.utils import is_tf_available, is_torch_available

from torch import nn, Tensor
from torch.nn import functional as F

import onnx
import onnxruntime


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
output = os.environ.get("ONNX_OUTPUT", f"{script_dir}/base_model")
output_path = Path(output)
output_path.mkdir(parents=True, exist_ok=True)

opset = int(os.environ.get("ONNX_OPSET"))
batch_size = int(os.environ.get("ONNX_BATCH_SIZE", "4"))
sequence_length = int(os.environ.get("ONNX_SEQUENCE_LENGTH", "256"))

model_id = "google-bert/bert-base-chinese"

# base_model = AutoModel.from_pretrained(model_id)
# onnx_path = Path("model.onnx")
# onnx_config_constructor = TasksManager.get_exporter_config_constructor("onnx", base_model)
# onnx_config = onnx_config_constructor(base_model.config)
# onnx_inputs, onnx_outputs = export(base_model, onnx_config, onnx_path, onnx_config.DEFAULT_ONNX_OPSET)


class CustomBertOnnxConfig(BertOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    ATOL_FOR_VALIDATION = 1e-4
    # DEFAULT_ONNX_OPSET = 14

    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {
            "input_ids": dynamic_axis,
            "attention_mask": dynamic_axis,
            "token_type_ids": dynamic_axis,
        }

class CustomDistilBertOnnxConfig(DistilBertOnnxConfig):
    # DEFAULT_ONNX_OPSET = 14
    @property
    def inputs(self) -> Dict[str, Dict[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch_size", 1: "num_choices", 2: "sequence_length"}
        else:
            dynamic_axis = {0: "batch_size", 1: "sequence_length"}
        return {"input_ids": dynamic_axis, "attention_mask": dynamic_axis}

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
onnx_config = CustomBertOnnxConfig(config=config, task="text-classification")
# onnx_config = DistilBertOnnxConfig(config, task="text-classification")
print(
    f"Model: {model_id}, inputs = {onnx_config.inputs}, outputs = {onnx_config.outputs}"
)

# base_model = AutoModelForSequenceClassification.from_pretrained(model_id)
# onnx_inputs, onnx_outputs = export(
#     model=base_model,
#     config=onnx_config,
#     output=onnx_path,
#     opset=opset,
#     device="cpu",
#     input_shapes={
#         "input_ids": [batch_size, sequence_length],
#         "attention_mask": [batch_size, sequence_length],
#         "token_type_ids": [batch_size, sequence_length],
#     },
#     dtype="fp32",
#     do_constant_folding=True,
#     # model_kwargs: Optional[Dict[str, Any]] = None,)
# )
# validate_model_outputs(
#     onnx_config, base_model, onnx_path, onnx_outputs, onnx_config.ATOL_FOR_VALIDATION
# )
# print("Model outputs are valid")
# print(
#     f"Model: {model_id} / {onnx_path}, opset = {opset}, inputs = {onnx_inputs}, outputs = {onnx_outputs}"
# )


# from optimum.exporters import utils ;  # _get_submodels_and_export_configs
custom_onnx_configs = {
    "model": onnx_config
}
main_export(
    model_name_or_path=model_id,
    output=output_path,
    task="text-classification",
    device="cpu",
    # dtype="fp32",
    # optimize="O1",
    opset=opset,
    framework="pt",
    pad_token_id=0,
    revision="main",
    custom_onnx_configs=custom_onnx_configs,
    # for_ort=True,
    # force_download=True,
    batch_size=batch_size,
    sequence_length=sequence_length,
    # kwargs_shapes={ # DEFAULT_DUMMY_SHAPES
    #     "batch_size": batch_size,
    #     "sequence_length": sequence_length,
    #     "num_choices": 4
    # },
    do_constant_folding=True,
    # model_kwargs: Optional[Dict[str, Any]] = None,)
)
print(f"Exported model to {output_path}")

