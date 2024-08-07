# First
The purpose of this project is to explore the application of Rust in NLP. On one hand, it is to become proficient in Rust programming; on the other hand, it serves as a starting point for learning NLP. The base model used in this article is `google-bert-chinese`. The training dataset is sourced from: `https://github.com/pengming617/bert_classification`. The implementation uses onnx-ort because the repository `git@github.com/ort.git` provides both train and predict examples. [Rust-bert](https://github.com/guillaume-be/rust-bert) does not have a train example and requires relying on a Python-trained model. Additionally, Python was not chosen because its capabilities are too limited, being only readable and runnable.

# Export google-bert-chinese, and artifacts
## Python, dir: onnx-model/google-bert-chinese
1. hfoptimum-model.py: export google-bert-chinese to dir `base_model`
1. hfoptimum-check.py: check base_model
    ```
    MODEL_PATH=onnx-model/google-bert-chinese/base_model \
        MODEL_FILE=model.onnx \
        AUTO_TOKENIZER_MODEL_NAME_OR_PATH=onnx-model/google-bert-chinese/base_model \
        python onnx-model/google-bert-chinese/hfoptimum-check.py
    ```
1. hfoptimum-glue.py: a example, from `git@github.com:huggingface/optimum.git` path = `examples/onnxruntime/optimization/text-classification/run_glue.py`
    ``` bash
    python onnx-model/google-bert-chinese/hfoptimum-glue.py \
        --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english \
        --task_name sst2 \
        --optimization_level 1 \
        --do_eval \
        --do_predict \
        --overwrite_output_dir \
        --output_dir /tmp/optimized_distilbert_sst2

    python onnx-model/google-bert-chinese/hfoptimum-glue.py \
            --model_name_or_path google-bert/bert-base-chinese \
            --task_name sst2 \
            --optimization_level 1 \
            --overwrite_output_dir \
            --output_dir onnx-model/google-bert-chinese/base_model
    ```
1. hfoptimum-model.py、hfoptimum-check.py、hfoptimum-glue.py require hfoptimum-requirements.txt
1. onnx-artifacts.py generate artifacts, output = onnx-artifacts. it require onnx-requirements.txt
1. make.sh is a shortcut script used for quickly executing the aforementioned Python code. But it depends on virtualenv and virtualenvwrapper.
    ```bash
    ./onnx-model/google-bert-chinese/make.sh
    ```
## Rust, using ort
### Depends on [ort=2.0.0-rc.4](https://crates.io/crates/ort/2.0.0-rc.4), [Guide](https://ort.pyke.io/)
1. Runtime depends on linux. I use OrbStack at macOS.
1. The path to the binary can be controlled with the environment variable `ORT_DYLIB_PATH=<project_path>/onnxruntime-libs/libonnxruntime.so.1.18.0`, [Source](https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz) [Releases](https://github.com/microsoft/onnxruntime/releases/tag/v1.18.0)
1. Runtime, LD_LIBRARY_PATH=<project_path>/onnxruntime-libs/
### onnx_bert_chinese_ort_train_dataset, from csv files to datasets
1. From `git@github.com:pykeio/ort.git, branch = 2.0.0-rc.4, file = examples/training/examples/pretokenize.rs`
1. Run 
    ```
    cargo run -r -p onnx_bert_chinese_ort_train_dataset -- \
        --tokenizer-json="./onnx-model/google-bert-chinese/base_model/tokenizer.json" \
        --files-chunks=1 \
        --out-dataset-bin="./target/dataset.bin" \
        --csv-delimiter="	" \
        --csvs="data/train.tsv"
    ```
### onnx_bert_chinese_ort_train, using datasets for train
1. From `From git@github.com:pykeio/ort.git, branch = 2.0.0-rc.4, file = examples/training/examples/train-clm.rs`
1. Run
    ```
    export RUST_LOG=train=trace,onnx_bert_chinese_ort_train=trace
    cargo run -p onnx_bert_chinese_ort_train -- \
        --bin-file-chunk-size=1 \
        --channel-buf-size=3 \
        --training-batch-size=4 \
        --training-sequence-length=256 \
        --optimizer-lr=7e-5 \
        --tokenizer-json="./onnx-model/google-bert-chinese/base_model/tokenizer.json" \
        --checkpoint-file="./onnx-model/google-bert-chinese/onnx-artifacts/checkpoint" \
        --training-model-file="./onnx-model/google-bert-chinese/onnx-artifacts/training_model.onnx" \
        --eval-model-file="./onnx-model/google-bert-chinese/onnx-artifacts/eval_model.onnx" \
        --optimizer-model-file="./onnx-model/google-bert-chinese/onnx-artifacts/optimizer_model.onnx" \
        --out-trained-onnx="./onnx-model/google-bert-chinese/onnx-artifacts/trained_model.onnx" \
        --dataset-bin="./target/dataset.bin/dataset-0-9600-1960.bin"
    ```

## The issue I am currently facing
1. ```
    # Run:
    cargo run -p onnx_bert_chinese_ort_train --example train

    Error: trainer.step(inputs, labels), error: Failed to run inference on model: /onnxruntime_src/orttraining/orttraining/training_api/module.cc:632 onnxruntime::common::Status onnxruntime::training::api::Module::TrainStep(const std::vector<OrtValue>&, std::vector<OrtValue>&) [ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Unexpected input data type. Actual: (tensor(int64)) , expected: (tensor(float))
    ```

# References
1. [onnxruntime](https://github.com/microsoft/onnxruntime)
1. [onnxruntime-training-examples](https://github.com/microsoft/onnxruntime-training-examples)
1. [onnxruntime.ai](https://onnxruntime.ai/)
1. [huggingface](https://huggingface.co/docs/transformers/serialization)
1. [Export a model to ONNX with optimum.exporters.onnx](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model)
1. [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)
1. [ORTModelForSequenceClassification](https://huggingface.co/docs/optimum/v1.2.1/en/onnxruntime/modeling_ort#optimum.onnxruntime.ORTModelForSequenceClassification)
1. [example-adding-support-for-bert](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/contribute#example-adding-support-for-bert)
1. [ort.pyke.io](https://ort.pyke.io/), [crate](https://crates.io/crates/ort)
