use std::{sync::Arc, time::Instant};

use futures::StreamExt;
use tokio::sync::RwLock;
use tracing::Level;

use onnx_ort_train_sentiment_dataset::{
    csv::{chunks_timeout_of_train_files, get_imported_lines},
    prelude::*,
    SinkDataset,
};
use yss_commons::commons_tokio::{block_on, setup_tracing};

/// cargo run -p onnx-ort-train-sentiment-dataset -- --csvs="./onnx-ort-train-sentiment-dataset/data/train.csv" --tokenizer-json="./tools/google-bert-chinese/model/tokenizer.json" --out-dataset-bin="./target/dataset.bin"
// #[tokio::main(flavor = "multi_thread", worker_threads = 4)]
fn main() -> anyhow::Result<()> {
    setup_tracing(
        &args().log_level,
        args().log_file.as_ref(),
        args().log_display_target,
    );
    info!(
        "OK. command args: {:?}, CWD={} ",
        args(),
        std::env::current_dir()?.display()
    );
    block_on(spawn_dataset_task())
}

#[tracing_attributes::instrument(level = Level::INFO, name = "spawn_dataset_task")]
async fn spawn_dataset_task() -> anyhow::Result<()> {
    let dataset_begin = Instant::now();
    let dataset_sink_writer = SinkDataset::new_by_args()?;
    let dataset_sink_writer = Arc::new(RwLock::new(dataset_sink_writer));
    let d = Box::pin(dataset_do(dataset_sink_writer.clone()));
    let r = tokio::spawn(d).await??;
    info!(
        "dataset cost: {:?}, total lines: {}",
        dataset_begin.elapsed(),
        get_imported_lines()
    );
    Ok(r)
}

async fn dataset_do(dataset_sink_writer: Arc<RwLock<SinkDataset>>) -> anyhow::Result<()> {
    let r = StreamExt::map(chunks_timeout_of_train_files(), Ok::<_, anyhow::Error>)
        .forward(SinkDataset::dataset_sink(dataset_sink_writer))
        .await;

    if let Err(ref e) = r {
        error!("Error convert to dataset: {}", e);
    }

    r
}
