use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Instant,
};

use futures::StreamExt;
use tokio::{sync::RwLock, task::JoinSet};
use tracing::Level;

use clap::{Parser, ValueHint};
use lazy_static::lazy_static;

use onnx_ort_train_sentiment_dataset::{csv::chunks_train_records, prelude::*, SinkDataset};
use yss_commons::commons_tokio::{block_on, setup_tracing};

#[derive(Parser, Debug)]
#[command(version, about)]
pub struct Args {
    /// csv files
    #[arg(short, long, value_parser, num_args = 1.., value_delimiter = ' ', required = true)]
    pub csvs: Vec<PathBuf>,

    #[clap(long, value_parser, value_hint = ValueHint::FilePath)]
    pub tokenizer_json: PathBuf,

    /// the parsent dir of dataset files, that from csv files. dataset bin files format is dataset-{}.bin
    #[clap(long, value_parser, value_hint = ValueHint::FilePath)]
    pub out_dataset_bin: PathBuf,

    /// csv delimiter, default = ,
    #[arg(long, default_value = ",")]
    pub csv_delimiter: char,

    #[arg(short = 'l', long, default_value = "INFO")]
    pub log_level: String,

    #[arg(long)]
    pub log_file: Option<PathBuf>,

    #[arg(long)]
    pub log_display_target: Option<bool>,

    /// At the same time, unordered files size. csv files is split to chunks by this size
    #[arg(long)]
    pub files_chunks: Option<usize>,

    /// It is max size of chunk, when import csv line
    #[arg(long, default_value = "100")]
    training_batch_size: usize,
}

lazy_static! {
    static ref COMMAND_ARGS: OnceLock<Args> = OnceLock::new();
}

// #[cfg(not(test))]
pub fn args() -> &'static Args {
    COMMAND_ARGS.get_or_init(Args::parse)
}

// // #[cfg(test)]
// pub fn args() -> &'static Args {
//     COMMAND_ARGS.get_or_init(|| {
//         Args::parse_from(vec!["program", "--csvs", "./onnx-ort-train-sentiment-dataset/data/train.csv"])
//     })
// }

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
    if out_dataset_bin.exists() {
        if !out_dataset_bin.is_dir() {
            panic!(
                "out_dataset_bin is not dir, it {}",
                out_dataset_bin.display()
            );
        }

        std::fs::remove_dir_all(&out_dataset_bin).inspect_err(|e| {
            panic!(
                "remove old dirs({}) error: {:?}",
                out_dataset_bin.display(),
                e
            );
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
    if args().files_chunks.is_none() || Some(0) == args().files_chunks {
        let t = inspect_do_by_files(0, args().csvs.to_vec()).await?;
        return Ok(t);
    }

    let mut tasks = args()
        .csvs
        .chunks(args().files_chunks.unwrap())
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
            Some(Ok(Ok(done))) => sum += done,
            Some(Ok(Err(e))) => {
                error!("dataset_do: {:?}", e);
                continue;
            }
            Some(Err(e)) => {
                error!("dataset_do: {:?}", e);
                break;
            }
        }
    }
    Ok(sum)
}

// 一小批文件，对应 一个dataset_sink_writer
#[tracing_attributes::instrument(level = Level::INFO, name = "inspect_dataset", skip(files))]
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
    let dataset_sink_writer = Arc::new(RwLock::new(
        SinkDataset::new_by_args(id, &args().tokenizer_json, &args().out_dataset_bin).await?,
    ));
    let _ = chunks_train_records(&files, args().csv_delimiter, args().training_batch_size)
        .take_while(|r| futures::future::ready(r.is_ok()))
        .forward(SinkDataset::dataset_sink(
            (&dataset_sink_writer).clone(),
            args().training_batch_size,
        ))
        .await?;

    let dataset = dataset_sink_writer.read().await;
    let total = dataset.total_lines();
    let ids_max_len = dataset.ids_max_len();
    let out = dataset.out_dataset_bin().to_path_buf();
    let mut out_with_lines = out.clone();
    out_with_lines.set_file_name(format!("dataset-{id}-{total}-{ids_max_len}.bin"));
    std::fs::rename(&out, &out_with_lines).inspect_err(|e| {
        error!(
            "rename {} to {} error: {:?}",
            out.display(),
            out_with_lines.display(),
            e
        );
    })?;

    info!(
        "Done(cost: {}). dataset: id={id}, total = {total}, ids_max_len = {ids_max_len}, bytes = {}, bin file = {}",
        dataset_begin.elapsed().as_secs_f32(),
        dataset.size_of_written(),
        out_with_lines.display()
    );
    anyhow::Result::Ok(total)
}
