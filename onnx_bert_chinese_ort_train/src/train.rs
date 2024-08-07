use std::{path::PathBuf, sync::OnceLock};

use anyhow::{anyhow, Ok};
use kdam::BarExt;
use onnx_bert_chinese_ort_train::{
    ort_training::{self},
    NextTraining,
};
use onnx_bert_chinese_ort_train_dataset::text_label::TextLabel;

use clap::{Parser, ValueHint};
use lazy_static::lazy_static;
use tokio::sync::mpsc;
use tracing::{debug, error, info, trace};
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
    pub bin_file_chunk_size: Option<usize>,

    /// It is max size of chunk, when import dataset items
    #[arg(long, default_value = "100")]
    training_batch_size: usize,

    /// default ids_max_len of files
    #[arg(long)]
    training_sequence_length: Option<usize>,

    #[arg(long, default_value = "10")]
    channel_buf_size: usize,
}

lazy_static! {
    static ref COMMAND_ARGS: OnceLock<Args> = OnceLock::new();
}

// #[cfg(not(test))]
pub fn args() -> &'static Args {
    COMMAND_ARGS.get_or_init(Args::parse)
}

fn main() -> anyhow::Result<()> {
    yoyo()
        .inspect(|_| info!("Good, Good, Good! training done!"))
        .inspect_err(|e| error!("Boom, Boom, Boom! training error: {:?}", e))?;
    Ok(())
}
fn yoyo() -> anyhow::Result<()> {
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

    args().dataset_bin.iter().fold(Ok(()), |e, v| {
        let _ = e?;
        if !v.exists() {
            return anyhow::Result::Err(anyhow!("dataset-bin not exists, it is {}", v.display()));
        }
        Ok(())
    })?;
    let out_trained_onnx_parsent = args().out_trained_onnx.parent().ok_or(anyhow!(
        "--out-trained-onnx error: {}",
        args().out_trained_onnx.display()
    ))?;
    if !out_trained_onnx_parsent.exists() {
        std::fs::create_dir_all(out_trained_onnx_parsent)?;
    }

    let channel = mpsc::channel::<Vec<TextLabel>>(args().channel_buf_size);
    block_on(training_do(channel))?;
    Ok(())
}

async fn training_do(
    (tx, mut rx): (mpsc::Sender<Vec<TextLabel>>, mpsc::Receiver<Vec<TextLabel>>),
) -> anyhow::Result<()> {
    // let ort_training = Arc::new(RwLock::new(ort_training));
    let next_training = NextTraining::new(
        args().dataset_bin.clone(),
        args().bin_file_chunk_size,
        args().training_batch_size,
        tx,
    )?;

    let training_steps = if next_training.total_records() % args().training_batch_size == 0 {
        next_training.total_records() / args().training_batch_size
    } else {
        next_training.total_records() / args().training_batch_size + 1
    };
    info!(
        "ort_training created, will train {} records, training_steps = {training_steps}, ids_max_len = {}",
        next_training.total_records(), next_training.ids_max_len()
    );

    // 没有实现Send
    let ort_training = ort_training::OrtTrainingBuilder::default()
        .with_checkpoint(&args().checkpoint_file)
        .with_training_model(&args().training_model_file)
        .with_eval_model(&args().eval_model_file)
        .with_optimizer_model(&args().optimizer_model_file)
        .with_tokenizer_json(&args().tokenizer_json)
        .with_out_trained_onnx(&args().out_trained_onnx)
        .with_optimizer_lr(args().optimizer_lr)
        .with_training_steps(training_steps)
        .with_training_batch_size(args().training_batch_size)
        .with_training_sequence_length(
            args()
                .training_sequence_length
                .unwrap_or(next_training.ids_max_len()),
        )
        .with_ids_max_len(next_training.ids_max_len())
        .build()
        .map_err(|e| anyhow!("create ort training fail, error = {e}"))?;

    let mut pb = kdam::tqdm!(total = training_steps);
    tokio::spawn(next_training.next_training());

    eprintln!();
    let count = 0_usize;
    while let Some(v) = rx.recv().await {
        let loss = ort_training
            .step(v.as_slice())
            .inspect_err(|e| {
                debug!("Step({count}) ort training step error, the record = {v:?}, error = {e}")
            })
            .inspect(|_| trace!("Step({count}) ort training step ok, the record = {v:?}"))?;
        pb.set_postfix(format!("Step({count})-loss={loss:.3}"));
        pb.update(1).unwrap();
    }
    eprintln!();
    let _ = kdam::term::show_cursor();
    info!("training done");

    ort_training.export()?;
    info!(
        "training done, export to {}",
        ort_training.out_trained_onnx().display()
    );

    Ok(())
}
