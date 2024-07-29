

onnx
https://github.com/microsoft/onnxruntime
https://onnxruntime.ai/
https://github.com/microsoft/onnxruntime-training-examples

huggingface
https://huggingface.co/docs/transformers/serialization

ort.pyke.io
https://ort.pyke.io/
https://crates.io/crates/ort

https://huggingface.co/docs/transformers/tflite
https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_CPU.ipynb


cargo run -r -p onnx-ort-train-sentiment-dataset -- \
    --tokenizer-json="./tools/google-bert-chinese/model/tokenizer.json" \
    --files-chunks=1 \
    --out-dataset-bin="./target/dataset.bin" \
    --csvs="./onnx-ort-train-sentiment-dataset/data/train.csv"

export RUST_LOG=train=trace,onnx_ort_train_sentiment=trace
cargo run -p onnx-ort-train-sentiment -- \
    --bin-file-chunk-size=1 \
    --channel-buf-size=3 \
    --training-batch-size=16 \
    --training-sequence-length=256 \
    --optimizer-lr=7e-5 \
    --tokenizer-json="./tools/google-bert-chinese/model/tokenizer.json" \
    --checkpoint-file="./tools/google-bert-chinese/onnx-training/checkpoint" \
    --training-model-file="./tools/google-bert-chinese/onnx-training/training_model.onnx" \
    --eval-model-file="./tools/google-bert-chinese/onnx-training/eval_model.onnx" \
    --optimizer-model-file="./tools/google-bert-chinese/onnx-training/optimizer_model.onnx" \
    --out-trained-onnx="./target/trained_model.onnx" \
    --dataset-bin="./target/dataset.bin/dataset-0-27-62.bin"

cargo run -p onnx-ort-train-sentiment -- \
    --bin-file-chunk-size=1 \
    --channel-buf-size=1 \
    --training-batch-size=1 \
    --optimizer-lr=7e-5 \
    --tokenizer-json="/home/yangss/workspace/rust/ort.git/examples/gpt2/data/tokenizer.json" \
    --checkpoint-file="/home/yangss/workspace/rust/ort.git/tools/train-data/mini-clm/checkpoint" \
    --training-model-file="/home/yangss/workspace/rust/ort.git/tools/train-data/mini-clm/training_model.onnx" \
    --eval-model-file="/home/yangss/workspace/rust/ort.git/tools/train-data/mini-clm/eval_model.onnx" \
    --optimizer-model-file="/home/yangss/workspace/rust/ort.git/tools/train-data/mini-clm/optimizer_model.onnx" \
    --out-trained-onnx="./target/trained_model.onnx" \
    --dataset-bin="./target/dataset.bin/dataset-0-27-62.bin"