use std::{path::PathBuf, sync::Arc, time::Instant};

use futures::StreamExt;
use tokio::{sync::RwLock, task::JoinSet};
use tracing::Level;

use onnx_ort_train_sentiment_dataset::{csv::chunks_train_records, prelude::*, SinkDataset};
use yss_commons::commons_tokio::{block_on, setup_tracing};

/// cargo run -p onnx-ort-train-sentiment-dataset -- --split-size=1 --out-dataset-bin="./target/dataset.bin --csvs="./onnx-ort-train-sentiment-dataset/data/train.csv" --tokenizer-json="./tools/google-bert-chinese/model/tokenizer.json""
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

    args().csvs.iter().for_each(|v| {
        if !v.exists() {
            panic!("csvs is empty, it {}", v.display());
        }
    });

    let out_dataset_bin = args().out_dataset_bin.as_path();
    if !out_dataset_bin.is_dir() {
        panic!("out_dataset_bin is not dir, it {}", out_dataset_bin.display());
    }
    if out_dataset_bin.exists() {
        std::fs::remove_dir_all(&out_dataset_bin).inspect_err(|e| {
            panic!("remove old dirs({}) error: {:?}", out_dataset_bin.display(), e);
        })?;
    }
    std::fs::create_dir_all(&out_dataset_bin).inspect_err(|e| {
        panic!("create dir({}) error: {:?}", out_dataset_bin.display(), e);
    })?;

    block_on(spawn_dataset_task())
}

#[tracing_attributes::instrument(level = Level::INFO, name = "spawn_dataset_task")]
async fn spawn_dataset_task() -> anyhow::Result<()> {
    let dataset_begin = Instant::now();
    let r = tokio::spawn(dataset_do()).await??;
    info!(
        "dataset cost: {:?}, total lines: {}",
        dataset_begin.elapsed(),
        r
    );
    Ok(())
}

async fn dataset_do() -> anyhow::Result<usize> {
    if args().split_size.is_none() || Some(0) == args().split_size {
        let t = inspect_do_by_files(0, args().csvs.to_vec()).await?;
        return Ok(t);
    }

    let mut tasks = args()
        .csvs
        .chunks(args().split_size.unwrap())
        .enumerate()
        .map(|(id, files)| inspect_do_by_files(id, files.to_vec()))
        .fold(JoinSet::new(), |mut set, fut| {
            set.spawn(fut);
            set
        });

    let mut sum = 0_usize;
    loop {
        let r = tasks.join_next().await;
        match r {
            None => break,
            Some(Ok(Ok(i))) => sum += i,
            Some(Err(e)) => {
                error!("dataset_do: {:?}", e);
                break;
            }
            Some(Ok(Err(e))) => {
                error!("dataset_do: {:?}", e);
                continue;
            }
        }
    }
    Ok(sum)
}

// 一小批文件，对应 一个dataset_sink_writer
#[tracing_attributes::instrument(level = Level::INFO, name = "inspect_do_by_files", skip(files))]
async fn inspect_do_by_files(id: usize, files: Vec<PathBuf>) -> anyhow::Result<usize> {
    let r = dataset_do_by_files(id, &files).await;
    match r {
        Ok(t) => {
            info!("inspect_do_by_files: id={}, total = {}", id, t);
            Ok(t)
        }
        Err(e) => {
            error!("inspect_do_by_files: id={}, error = {:?}", id, e);
            Err(e)
        }
    }
}
async fn dataset_do_by_files(id: usize, files: &[PathBuf]) -> anyhow::Result<usize> {
    let dataset_begin = Instant::now();
    info!("Begin. dataset: id={}, files={:?}", id, files);
    let dataset_sink_writer = Arc::new(RwLock::new(SinkDataset::new_by_args(id)?));
    let _ = chunks_train_records(&files)
        .map(|r| anyhow::Result::Ok(r))
        .forward(SinkDataset::dataset_sink((&dataset_sink_writer).clone()))
        .await?;

    let dataset = dataset_sink_writer.read().await;
    let total = dataset.total_lines();
    info!(
        "Done(cost: {}). dataset: id={}, total = {}, bytes = {}, bin file = {}",
        dataset_begin.elapsed().as_secs_f32(),
        id,
        total,
        dataset.size_of_written(),
        dataset.out_dataset_bin().display()
    );
    anyhow::Result::Ok(total)
}
