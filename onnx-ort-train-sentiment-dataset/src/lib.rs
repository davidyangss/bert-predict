pub mod csv;

use std::{
    fs::OpenOptions,
    future::Future,
    io::IoSlice,
    path::PathBuf,
    sync::{atomic::AtomicUsize, Arc, OnceLock},
};

use anyhow::Ok;
use clap::{Parser, ValueHint};
use csv::Record;
use futures::{
    sink::{self},
    stream, Sink, StreamExt, TryStreamExt,
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

    /// At the same time, unordered files size.
    #[arg(long)]
    pub unordered: Option<usize>,

    /// It is max size of chunk, when import csv line
    #[arg(long, default_value = "1000")]
    chunk_max_size: usize,
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

type TextLabel = (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>);

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

    pub fn dataset_sink(
        dataset: Arc<RwLock<SinkDataset>>,
    ) -> impl Sink<Vec<Record>, Error = anyhow::Error> + Unpin {
        let sink = sink::unfold(dataset, |s, records| async move {
            Self::write_tokinizer_encode(&s.clone(), records).await?;
            Ok(s)
        });
        Box::pin(sink)
    }

    fn encode_to_bytes(
        dataset: &Arc<RwLock<Self>>,
        r: Record,
    ) -> impl Future<Output = anyhow::Result<TextLabel>> + 'static {
        let dataset = dataset.clone();
        async move {
            let dataset = dataset.read().await;

            let id_bytes: Vec<u8> = dataset
                .tokinizer
                .encode(r.comment(), false)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .get_ids()
                .iter()
                .flat_map(|c| (*c as u16).to_le_bytes())
                .collect();
            let id_len = id_bytes.len().to_le_bytes().to_vec();
            let label_bytes: Vec<u8> = dataset
                .tokinizer
                .encode(r.sentiment().to_string(), false)
                .map_err(|e| anyhow::anyhow!("{:?}", e))?
                .get_ids()
                .iter()
                .flat_map(|c| (*c as u16).to_le_bytes())
                .collect();
            let label_len = label_bytes.len().to_le_bytes().to_vec();
            Ok((id_len, id_bytes, label_len, label_bytes))
        }
    }

    fn write_bytes_to_dataset(
        dataset: &Arc<RwLock<Self>>,
        r: Vec<anyhow::Result<TextLabel>>,
    ) -> impl Future<Output = anyhow::Result<()>> + 'static {
        let dataset = dataset.clone();
        async move {
            // 写文件
            let mut dataset = dataset.write().await;
            let bufs: anyhow::Result<Vec<IoSlice>> = r
                .iter() // Iter item Result时，遇到第一个Error停止
                .fold(
                    Ok(Vec::<IoSlice>::new()),
                    |acc: anyhow::Result<Vec<IoSlice>>, r: &anyhow::Result<TextLabel>| {
                        if let Err(_) = acc {
                            return acc;
                        }

                        match r {
                            Err(e) => return Err(anyhow::anyhow!("{}", e)),
                            Result::Ok(vv) => {
                                let mut acc = acc.unwrap();
                                acc.push(IoSlice::new(vv.0.as_slice()));
                                acc.push(IoSlice::new(vv.1.as_slice()));
                                acc.push(IoSlice::new(vv.2.as_slice()));
                                acc.push(IoSlice::new(vv.3.as_slice()));
                                Ok(acc)
                            }
                        }
                    },
                );
            if let Err(e) = bufs {
                return Err(e);
            }

            let writer = &mut dataset.writer;
            let writed: Result<usize, std::io::Error> =
                writer.write_vectored(bufs.unwrap().as_slice()).await;
            if let Err(e) = writed {
                return Err(anyhow::anyhow!(
                    "write_tokinizer_encode write error: {:?}",
                    e
                ));
            }
            if let Err(e) = writer.flush().await {
                return Err(anyhow::anyhow!(
                    "write_tokinizer_encode flush error: {:?}",
                    e
                ));
            }
            if let Result::Ok(u) = writed {
                dataset
                    .size_of_written
                    .fetch_add(u, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(())
        }
    }

    async fn write_tokinizer_encode(
        dataset: &Arc<RwLock<Self>>, // 目测，不清楚tokinizer是否是线程安全的
        records: Vec<Record>,
    ) -> anyhow::Result<()> {
        let s = stream::iter(records)
            .then(|r| Self::encode_to_bytes(dataset, r))
            .inspect_err(|e| error!("tokinizer encode error: {:?}", e))
            .ready_chunks(args().chunk_max_size)
            .then(|r| Self::write_bytes_to_dataset(dataset, r))
            .inspect_err(|e| error!("tokinizer write error: {:?}", e))
            .take_while(|r| std::future::ready(r.is_ok())); // 有错误就停止
        s.for_each_concurrent(num_cpus::get() * 3,|_| async {}).await;
        Ok(())
        // warn!("Dataset.bin is written size: {}", dataset.size_of_written.load(std::sync::atomic::Ordering::SeqCst));
    }
}
