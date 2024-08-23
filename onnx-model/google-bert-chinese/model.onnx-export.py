# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import torch
import numpy as np
import onnx
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from torch import nn, Tensor
from torch.nn import functional as F


batch_size = 4
max_seq_length = 256
opset_version = 17


def build_base_model(tokenizer, model_dir, device):
    # model_path = os.path.join(model_dir, "pytorch_model.bin")
    model_path = model_dir
    config_path = os.path.join(model_dir, "config.json")
    
    config_kwargs = {
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
    }
    config = AutoConfig.from_pretrained(config_path, **config_kwargs)
    # config_dict = config.to_dict()
    # for key, value in config_dict.items():
    #     print(f"{key}: {value}")
    # print(f"model_path = {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, config=config, revision="main", use_auth_token=None
    )
    # for name, param in model.named_parameters():
    #     print(f"{name}: {param.size()}")
    model.to(device=device)

    # set the model to inference mode
    # It is important to call torch_model.eval() or torch_model.train(False) before exporting the model,
    # to turn the model to inference mode. This is required since operators like dropout or batchnorm
    # behave differently in inference and training mode.
    model.eval()
    model.resize_token_embeddings(len(tokenizer))
    return model


def build_tokenizer(tokenizer_name):
    tokenizer_kwargs = {
        "cache_dir": None,
        "use_fast": True,
        "revision": "main",
        "use_auth_token": None,
    }
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)
    return tokenizer


class RefineModel(torch.nn.Module):
    def __init__(self, tokenizer, model_dir, device="cpu"):
        super(RefineModel, self).__init__()
        self._base_model = build_base_model(tokenizer, model_dir, device)

    # def forward(self, input_ids, attention_mask, token_type_ids):
    #     x = self._base_model(input_ids, attention_mask, token_type_ids)
    #     return x[0].argmax(dim=-1)
    # def forward(self, input_ids, attention_mask, token_type_ids):
    #     x = self._base_model(input_ids, attention_mask, token_type_ids)
    #     return x.logits.float()
    # def forward(self, input_ids, attention_mask, token_type_ids) -> Tensor:
    #     x = self._base_model(input_ids, attention_mask, token_type_ids)
    #     logits = x.logits
    #     return logits.view(-1, logits.size(-1))
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self._base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # logits = outputs.logits.to(torch.float32)
        # return logits.view(-1, logits.size(-1))
        return outputs


def export_onnx(model_dir, save_path, seq_len=256, batch_size=16):
    # build tokenizer
    tokenizer = build_tokenizer(tokenizer_name=model_dir)

    # build model
    model = RefineModel(tokenizer, model_dir)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    # Generate dummy inputs to the model. Adjust if necessary.
    inputs = {
        "input_ids": torch.randint(max_seq_length, [batch_size, max_seq_length], dtype=torch.int64).to(
            device
        ),  # list of numerical ids for the tokenised text
        "attention_mask": torch.ones([batch_size, max_seq_length], dtype=torch.int64).to(
            device
        ),  # dummy list of ones
        "token_type_ids": torch.ones([batch_size, max_seq_length], dtype=torch.int64).to(
            device
        ),  # dummy list of ones
    }
    symbolic_names = {0: "batch_size", 1: "max_seq_len"}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.onnx.export(
        model,  # model being run
        (
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
        ),  # model input (or a tuple for multiple inputs)
        f=save_path,  # where to save the model (can be a file or file-like object)
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ],  # the model's input names
        output_names=["logits"],  # the model's output names
        dynamic_axes={
            "input_ids": symbolic_names,
            "attention_mask": symbolic_names,
            "token_type_ids": symbolic_names,
            "logits": {0: "batch_size"}
        },
    )  # variable length axes

if __name__ == "__main__":
    model_dir = sys.argv[1]
    save_path = sys.argv[2]
    seq_len = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    export_onnx(model_dir, save_path, seq_len, batch_size)

    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
