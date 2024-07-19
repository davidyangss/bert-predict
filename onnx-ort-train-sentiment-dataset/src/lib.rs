pub mod csv;

use std::{
    fs::OpenOptions,
    io::IoSlice,
    path::PathBuf,
    sync::{atomic::AtomicUsize, Arc, OnceLock},
};

use anyhow::Ok;
use clap::{Parser, ValueHint};
use csv::Record;
use futures::{
    sink::{self},
    stream, Sink, StreamExt,
};
use lazy_static::lazy_static;

pub mod prelude {
    pub use crate::args;

    pub use tracing::debug;
    pub use tracing::error;
    pub use tracing::info;
    pub use tracing::trace;
    pub use tracing::warn;
}

use prelude::*;
use tokenizers::Tokenizer;
use tokio::{
    io::{AsyncWriteExt, BufWriter},
    sync::RwLock,
};

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct Args {
    /// csv files
    #[arg(short, long, value_parser, num_args = 1.., value_delimiter = ' ', required = true)]
    pub csvs: Vec<PathBuf>,

    #[clap(long, value_parser, value_hint = ValueHint::FilePath)]
    pub tokenizer_json: PathBuf,

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

    /// At the same time, unordered files size. default: 20
    #[arg(long, default_value = "20")]
    pub unordered: usize,

    /// It is max size of chunk, when import csv line
    #[arg(long, default_value = "1000")]
    chunk_max_size: usize,

    /// It is timeout of chunk, when import csv line. unit: milliseconds
    #[arg(long, default_value = "500")]
    chunk_timeout: u64,
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

type TextLabel = (Vec<u8>, Vec<u8>);

pin_project_lite::pin_project! {
    pub struct SinkDataset {
        tokinizer: Tokenizer,
        out_dataset_bin: PathBuf,
        #[pin]
        writer: BufWriter<tokio::fs::File>,
        size_of_written: AtomicUsize,
    }
}

impl SinkDataset {
    pub fn new_by_args() -> Result<Self, anyhow::Error> {
        let tokinizer = Tokenizer::from_file(&args().tokenizer_json).unwrap();
        let out_dataset_bin = args().out_dataset_bin.clone();
        let p = out_dataset_bin.parent().expect(&format!(
            "cannot get parent path of {}",
            &out_dataset_bin.display()
        ));
        if !p.exists() {
            std::fs::create_dir_all(p).inspect_err(|e| {
                error!("create dir({}) error: {:?}", p.display(), e);
            })?;
        }
        let f = OpenOptions::new()
            .append(false)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&out_dataset_bin)
            .inspect_err(|e| {
                error!("Open file {} error: {:?}", out_dataset_bin.display(), e);
            })?;
        let f = tokio::fs::File::from_std(f);
        let writer = BufWriter::new(f);
        let size_of_written = AtomicUsize::new(0);
        Ok(Self {
            tokinizer,
            out_dataset_bin,
            writer,
            size_of_written,
        })
    }

    pub fn dataset_sink(self) -> impl Sink<Vec<Record>, Error = anyhow::Error> + Unpin {
        let sink = sink::unfold(Arc::new(RwLock::new(self)), |s, records| async move {
            Self::write_tokinizer_encode(s.clone(), records).await;
            Ok(s)
        });
        Box::pin(sink)
    }

    async fn write_tokinizer_encode(
        dataset: Arc<RwLock<Self>>, // 目测，不清楚tokinizer是否是线程安全的
        records: Vec<Record>,
    ) {
        let dataset = &dataset.clone();
        let s = stream::iter(records)
            .then(|r| async move {
                // 将encode拆分为很多个小任务
                let dataset = dataset.clone();
                let dataset = dataset.read().await;
                let tokenized = dataset.tokinizer.encode(r.comment(), false);
                if let Err(e) = tokenized {
                    return Result::<TextLabel, anyhow::Error>::Err(anyhow::anyhow!(
                        "tokenized error: {:?}",
                        e
                    ));
                }

                let id_bytes: Vec<u8> = tokenized
                    .unwrap()
                    .get_ids()
                    .iter()
                    .flat_map(|c| (*c as u16).to_le_bytes())
                    .collect();
                Ok((id_bytes, vec![]))
            })
            .filter_map(|r| async { //过滤
                match r {
                    Result::Ok(r) => Some(r),
                    Err(e) => {
                        error!("write_tokinizer_encode error: {:?}", e);
                        None
                    }
                }
            })
            .ready_chunks(args().chunk_max_size)
            .for_each(|r| async move {  // 写文件
                let dataset = dataset.clone();
                let mut dataset = dataset.write().await;
                let bufs: Vec<IoSlice> = r
                    .iter()
                    .flat_map(|(id_bytes, label_bytes)| {
                        vec![
                            IoSlice::new(id_bytes.as_slice()),
                            IoSlice::new(label_bytes.as_slice()),
                        ]
                    })
                    .collect();
                let writer = &mut dataset.writer;
                if let Err(e) = writer.write_vectored(bufs.as_slice()).await {
                    error!("write_tokinizer_encode write error: {:?}", e);
                }
                if let Err(e) = writer.flush().await {
                    error!("write_tokinizer_encode flush error: {:?}", e);
                }
            });
        s.await;
    }
}
