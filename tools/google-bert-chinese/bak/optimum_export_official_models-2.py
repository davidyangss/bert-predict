from optimum.exporters.onnx import main_export
from optimum.exporters.onnx.model_configs import BertOnnxConfig
from transformers import AutoConfig

from optimum.exporters.onnx.base import ConfigBehavior
from typing import Dict
import torch

class CustomBertOnnxConfig(BertOnnxConfig):
    def __init__(self, config, task, opset_version=14):
        super().__init__(config, task)
        self.opset_version = opset_version

    def inputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch", 1: "seq"}
        }

    def outputs(self) -> Dict[str, Dict[int, str]]:
        return {
            "probs": {0: "batch", 1: "batch_seq"}
        }

    def dynamic_axes(self) -> Dict[str, Dict[int, str]]:
        return {
            "input_ids": {0: "batch", 1: "seq"},
            "probs": {0: "batch", 1: "batch_seq"}
        }

    def generate_dummy_inputs(self, batch_size=1, seq_len=64):
        return {"input_ids": torch.randint(0, self.config.vocab_size, (batch_size, seq_len))}

    def torch_to_onnx_output_map(self):
        return {}

model_id = "bert-base-chinese"
config = AutoConfig.from_pretrained(model_id)

custom_bert_onnx_config = CustomBertOnnxConfig(
    config=config,
    task="text-classification",
    opset_version=14
)

custom_onnx_configs = {
    "default": custom_bert_onnx_config
}

main_export(
    model_id,
    output="/home/yangss/workspace/rust/predict_docs.git/tools/bert-chinese/model/model-2.onnx",
    no_post_process=True,
    model_kwargs={"output_attentions": True},
    custom_onnx_configs=custom_bert_onnx_config
)
