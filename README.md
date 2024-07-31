

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
    --tokenizer-json="./tools/google-bert-chinese/base_model/tokenizer.json" \
    --files-chunks=1 \
    --out-dataset-bin="./target/dataset.bin" \
    --csvs="./onnx-ort-train-sentiment-dataset/data/train.csv"

export RUST_LOG=train=trace,onnx_ort_train_sentiment=trace
cargo run -p onnx-ort-train-sentiment -- \
    --bin-file-chunk-size=1 \
    --channel-buf-size=3 \
    --training-batch-size=4 \
    --training-sequence-length=256 \
    --optimizer-lr=7e-5 \
    --tokenizer-json="./tools/google-bert-chinese/base_model/tokenizer.json" \
    --checkpoint-file="./tools/google-bert-chinese/onnx-artifacts/checkpoint" \
    --training-model-file="./tools/google-bert-chinese/onnx-artifacts/training_model.onnx" \
    --eval-model-file="./tools/google-bert-chinese/onnx-artifacts/eval_model.onnx" \
    --optimizer-model-file="./tools/google-bert-chinese/onnx-artifacts/optimizer_model.onnx" \
    --out-trained-onnx="./tools/google-bert-chinese/onnx-artifacts/trained_model.onnx" \
    --dataset-bin="./target/dataset.bin/dataset-0-27-60.bin"