use std::time::Instant;

use futures::StreamExt;
use tokio::task;
use tracing::Level;

use onnx_ort_train_sentiment_dataset::{
    csv::chunks_timeout_of_train_files, prelude::*, SinkDataset,
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
    let r = task::spawn(dataset_do()).await?;
    info!("dataset cost: {:?},", dataset_begin.elapsed());
    r
}

// fn dataset_sink() -> impl Sink<Vec<Record>, Error = anyhow::Error> + Unpin {
//     let sink = sink::unfold(0, |mut sum, records: Vec<Record>| async move {
//         let t = task::spawn(async move {
//             sum += records.len();
//             info!("records: {:?}, dataset items: {}", &records, sum);
//             info!("dataset items: {} / {}", get_imported_lines(), sum);

//             Ok::<usize, anyhow::Error>(sum)
//         })
//         .await??;
//         Ok(t)
//     });
//     Box::pin(sink)
// }

async fn dataset_do() -> anyhow::Result<()> {
    let r = StreamExt::map(chunks_timeout_of_train_files(), Ok::<_, anyhow::Error>)
        .forward(SinkDataset::new_by_args().unwrap().dataset_sink())
        .await;

    if let Err(ref e) = r {
        error!("Error convert to dataset: {}", e);
    }

    r
}
