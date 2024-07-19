use onnx_ort_train_sentiment_dataset::prelude::*;

// #[tokio::main(flavor = "multi_thread", worker_threads = 4)]
fn main() -> anyhow::Result<()> {
    init();
    block_on(spawn_dataset_task())
}
