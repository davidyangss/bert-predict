use std::{
    fs::File,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, OnceLock, RwLock},
    time::Instant,
};

use anyhow::{anyhow, Ok};
use kdam::BarExt;
use ndarray::{concatenate, s, Array1, Array2, ArrayViewD, Axis};
use onnx_ort_train_sentiment::ort_training;
use onnx_ort_train_sentiment::Training;
use ort::{Allocator, CUDAExecutionProvider, Checkpoint, Session, SessionBuilder, Trainer};
use tokenizers::Tokenizer;

use clap::{Parser, ValueHint};
use lazy_static::lazy_static;
use tokio::task::JoinSet;
use tracing::{error, info, Level};
use yss_commons::commons_tokio::*;

#[derive(Parser, Debug)]
#[command(version, about)]
pub struct Args {
    /// pretrained dataset bin files
    #[arg(short = 'f', long, value_parser, num_args = 1.., value_delimiter = ' ', required = true)]
    pub dataset_bin: Vec<PathBuf>,

    /// tokenizer json file, example: examples/gpt2/data/tokenizer.json
    #[arg(long, value_parser, value_hint = ValueHint::FilePath, required = true)]
    pub tokenizer_json: PathBuf,

    /// Trainer checkpoint file, example: tools/train-data/mini-clm/checkpoint
    #[arg(long, value_parser, value_hint = ValueHint::FilePath, required = true)]
    pub checkpoint_file: PathBuf,

    /// Training model file, example: tools/train-data/mini-clm/training_model.onnx
    #[arg(long, value_parser, value_hint = ValueHint::FilePath, required = true)]
    pub training_model_file: PathBuf,

    /// Eval model file, example: tools/train-data/mini-clm/eval_model.onnx
    #[arg(long, value_parser, value_hint = ValueHint::FilePath, required = true)]
    pub eval_model_file: PathBuf,

    /// Optimizer model file, example: tools/train-data/mini-clm/optimizer_model.onnx
    #[arg(long, value_parser, value_hint = ValueHint::FilePath, required = true)]
    pub optimizer_model_file: PathBuf,

    #[arg(long, value_parser, default_value = "7e-5")]
    pub optimizer_lr: f32,

    #[arg(long, value_parser, value_hint = ValueHint::FilePath, required = true)]
    pub out_trained_onnx: PathBuf,

    #[arg(short = 'l', long, default_value = "INFO")]
    pub log_level: String,

    #[arg(long)]
    pub log_file: Option<PathBuf>,

    #[arg(long)]
    pub log_display_target: Option<bool>,

    /// At the same time, unordered files size. dataset.bin files is split to chunks by this size
    #[arg(long)]
    pub bin_chunks: Option<usize>,

    /// It is max size of chunk, when import dataset items
    #[arg(long, default_value = "100")]
    chunk_max_size: usize,
}

lazy_static! {
    static ref COMMAND_ARGS: OnceLock<Args> = OnceLock::new();
}

// #[cfg(not(test))]
pub fn args() -> &'static Args {
    COMMAND_ARGS.get_or_init(Args::parse)
}

// cargo run -p onnx-ort-train-sentiment -- --bin-chunks=1 --optimizer-lr=7e-5 --tokenizer-json="./tools/google-bert-chinese/model/tokenizer.json" --checkpoint-file="./tools/google-bert-chinese/onnx-training/checkpoint" --training-model-file="./tools/google-bert-chinese/onnx-training/training_model.onnx" --eval-model-file="./tools/google-bert-chinese/onnx-training/eval_model.onnx" --optimizer-model-file="./tools/google-bert-chinese/onnx-training/optimizer_model.onnx" --out-trained-onnx="./target/trained_model.onnx" --dataset-bin="./target/dataset.bin/dataset-0.bin"
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

    args().dataset_bin.iter().for_each(|v| {
        if !v.exists() {
            panic!("dataset-bin not exists, it {}", v.display());
        }
    });
    let out_trained_onnx_parsent = args().out_trained_onnx.parent().ok_or(anyhow!(
        "--out-trained-onnx error: {}",
        args().out_trained_onnx.display()
    ))?;
    if !out_trained_onnx_parsent.exists() {
        std::fs::create_dir_all(out_trained_onnx_parsent)?;
    }

    let ort_training = ort_training::OrtTrainingBuilder::default()
        .with_checkpoint(&args().checkpoint_file)
        .with_training_model(&args().training_model_file)
        .with_eval_model(&args().eval_model_file)
        .with_optimizer_model(&args().optimizer_model_file)
        .with_tokenizer_json(&args().tokenizer_json)
        .with_out_trained_onnx(&args().out_trained_onnx)
        .with_optimizer_lr(args().optimizer_lr)
        .build()?;
    info!("ort_training created");

    let ort_training = Arc::new(RwLock::new(ort_training));
    let training = Training::new(
        args().dataset_bin.clone(),
        args().bin_chunks,
        args().chunk_max_size,
        ort_training,
    )?;
    info!("ort_training created, will train {}", training.total_records());

    block_on(training.spawn_training_task())
}
