# https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model#customize-the-export-of-transformers-models-with-custom-modeling
from optimum.exporters.onnx import main_export

from transformers import AutoConfig

from optimum.exporters.onnx.config import TextDecoderOnnxConfig
from optimum.utils import NormalizedTextConfig, DummyPastKeyValuesGenerator
from typing import Dict


class MPTDummyPastKeyValuesGenerator(DummyPastKeyValuesGenerator):
    """
    MPT swaps the two last dimensions for the key cache compared to usual transformers
    decoder models, thus the redefinition here.
    """
    def generate(self, input_name: str, framework: str = "pt"):
        past_key_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            self.sequence_length,
        )
        past_value_shape = (
            self.batch_size,
            self.num_attention_heads,
            self.sequence_length,
            self.hidden_size // self.num_attention_heads,
        )
        return [
            (
                self.random_float_tensor(past_key_shape, framework=framework),
                self.random_float_tensor(past_value_shape, framework=framework),
            )
            for _ in range(self.num_layers)
        ]

class CustomMPTOnnxConfig(TextDecoderOnnxConfig):
    DUMMY_INPUT_GENERATOR_CLASSES = (MPTDummyPastKeyValuesGenerator,) + TextDecoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES
    DUMMY_PKV_GENERATOR_CLASS = MPTDummyPastKeyValuesGenerator

    DEFAULT_ONNX_OPSET = 14  # aten::tril operator requires opset>=14
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig.with_args(
        hidden_size="d_model",
        num_layers="n_layers",
        num_attention_heads="n_heads"
    )

    def add_past_key_values(self, inputs_or_outputs: Dict[str, Dict[int, str]], direction: str):
        """
        Adapted from https://github.com/huggingface/optimum/blob/v1.9.0/optimum/exporters/onnx/base.py#L625
        """
        if direction not in ["inputs", "outputs"]:
            raise ValueError(f'direction must either be "inputs" or "outputs", but {direction} was given')

        if direction == "inputs":
            decoder_sequence_name = "past_sequence_length"
            name = "past_key_values"
        else:
            decoder_sequence_name = "past_sequence_length + 1"
            name = "present"

        for i in range(self._normalized_config.num_layers):
            inputs_or_outputs[f"{name}.{i}.key"] = {0: "batch_size", 3: decoder_sequence_name}
            inputs_or_outputs[f"{name}.{i}.value"] = {0: "batch_size", 2: decoder_sequence_name}


model_id = "/home/fxmarty/hf_internship/optimum/tiny-mpt-random-remote-code"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

onnx_config = CustomMPTOnnxConfig(
    config=config,
    task="text-generation",
    use_past_in_inputs=False,
    use_present_in_outputs=True,
)
onnx_config_with_past = CustomMPTOnnxConfig(config, task="text-generation", use_past=True)

custom_onnx_configs = {
    "decoder_model": onnx_config,
    "decoder_with_past_model": onnx_config_with_past,
}

main_export(
    model_id,
    output="mpt_onnx",
    task="text-generation-with-past",
    trust_remote_code=True,
    custom_onnx_configs=custom_onnx_configs,
    no_post_process=True,
)