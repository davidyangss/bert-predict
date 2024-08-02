

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
    --save_strategy steps \
    --save_steps 100 \
    --use_cpu true \
    --fp16 true \
    --label_names 0 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 7e-5 \
    --num_train_epochs 3 \
    --max_steps 3 \
    --max_seq_length 256 \
    --overwrite_output_dir \
    --output_dir onnx-model/google-bert-chinese/base_model/

python onnx-model/google-bert-chinese/hfoptimum-glue.py \
    --model_name_or_path google-bert/bert-base-chinese \
    --task_name sst2 \
    --optimization_level 1 \
    --overwrite_output_dir \
    --output_dir onnx-model/google-bert-chinese/base_model/


cargo run -r -p onnx-ort-train-sentiment-dataset -- \
    --tokenizer-json="./onnx-model/google-bert-chinese/base_model/tokenizer.json" \
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
    --tokenizer-json="./onnx-model/google-bert-chinese/base_model/tokenizer.json" \
    --checkpoint-file="./onnx-model/google-bert-chinese/onnx-artifacts/checkpoint" \
    --training-model-file="./onnx-model/google-bert-chinese/onnx-artifacts/training_model.onnx" \
    --eval-model-file="./onnx-model/google-bert-chinese/onnx-artifacts/eval_model.onnx" \
    --optimizer-model-file="./onnx-model/google-bert-chinese/onnx-artifacts/optimizer_model.onnx" \
    --out-trained-onnx="./onnx-model/google-bert-chinese/onnx-artifacts/trained_model.onnx" \
    --dataset-bin="./target/dataset.bin/dataset-0-27-60.bin"