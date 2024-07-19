use onnx_ort_train_sentiment_dataset::prelude::*;
use yss_commons::commons_tokio::block_on;

// #[tokio::main(flavor = "multi_thread", worker_threads = 4)]
fn main() -> anyhow::Result<()> {
    init();
    block_on(spawn_dataset_task())
}
