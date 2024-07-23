#![feature(coroutines, proc_macro_hygiene, stmt_expr_attributes)]
#![feature(cursor_remaining)]

pub mod csv;
pub mod text_label;

use std::{
    fs::OpenOptions,
    future::Future,
    path::{Path, PathBuf},
    sync::{atomic::AtomicUsize, Arc},
};

use anyhow::Ok;
use csv::Record;
use futures::{
    sink::{self},
    stream, Sink, StreamExt, TryStreamExt,
};

pub mod prelude {

    pub use tracing::debug;
    pub use tracing::error;
    pub use tracing::info;
    pub use tracing::trace;
    pub use tracing::warn;
}

use prelude::*;
use text_label::{DatasetWriter, TextLabel};
use tokenizers::Tokenizer;
use tokio::{fs::File, io::AsyncWriteExt, sync::RwLock};

pin_project_lite::pin_project! {
    pub struct SinkDataset {
        id: usize,
        tokinizer: Tokenizer,
        out_dataset_bin: PathBuf,
        #[pin]
        writer: DatasetWriter<TextLabel, File>,
        total_lines: AtomicUsize,
    }
}

impl SinkDataset {
    pub async fn new_by_args(
        id: usize,
        tokenizer_json: &PathBuf,
        out_dataset_bin: &PathBuf,
    ) -> Result<Self, anyhow::Error> {
        let tokinizer = Tokenizer::from_file(tokenizer_json.clone()).unwrap();
        let out_dataset_bin = out_dataset_bin.clone();
        let out_dataset_bin = out_dataset_bin.join(format!("dataset-{}.bin", id));
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
        let writer = DatasetWriter::new(f, TextLabel::v1_style()).await?;
        Ok(Self {
            id,
            tokinizer,
            out_dataset_bin,
            writer,
            total_lines: AtomicUsize::new(0),
        })
    }

    pub fn total_lines(&self) -> usize {
        self.total_lines.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn size_of_written(&self) -> usize {
        self.writer.size_of_written()
    }
    pub fn lines_plus_plus(&self) -> usize {
        self.total_lines
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
    pub fn out_dataset_bin(&self) -> &Path {
        &self.out_dataset_bin
    }

    pub fn dataset_sink(
        dataset: Arc<RwLock<SinkDataset>>,
        chunk_max_size: usize,
    ) -> impl Sink<Vec<Record>, Error = anyhow::Error> + Unpin {
        let sink = sink::unfold(dataset, move |s, records| async move {
            Self::write_tokinizer_encode(&s.clone(), records, chunk_max_size).await?;
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
            dataset.lines_plus_plus();
            let id_bytes = dataset.writer.item_style().tokenizer_ids_as_bytes(
                &dataset
                    .tokinizer
                    .encode(r.comment(), false)
                    .map_err(|e| {
                        anyhow::anyhow!("tokinizer encode to bytes, error:{}", e.to_string())
                    })?
                    .get_ids(),
            );
            trace!(
                "tokinizer comment <{}> to {:X} / [{:?}]",
                r.comment(),
                id_bytes.len(),
                hex::encode(&id_bytes)
            );
            let label_bytes = dataset.writer.item_style().tokenizer_ids_as_bytes(
                &dataset
                    .tokinizer
                    .encode(r.sentiment().to_string(), false)
                    .map_err(|e| {
                        anyhow::anyhow!("tokinizer encode to bytes, error:{}", e.to_string())
                    })?
                    .get_ids(),
            );
            trace!(
                "tokinizer sentiment <{}> to {} / [{:?}]",
                r.sentiment(),
                label_bytes.len(),
                hex::encode(&label_bytes)
            );
            Ok(dataset.writer.style_bytes(id_bytes, label_bytes))
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
            let bufs = r
                .into_iter() // Iter item Result时，遇到第一个Error停止
                .fold(
                    Ok(Vec::<TextLabel>::new()),
                    |acc: anyhow::Result<Vec<TextLabel>>, r: anyhow::Result<TextLabel>| {
                        if let Err(_) = acc {
                            return acc;
                        }

                        match r {
                            Err(e) => Err(anyhow::anyhow!(
                                "write_tokinizer_encode write error: {}",
                                e.to_string()
                            )),
                            Result::Ok(vv) => Ok({
                                let mut acc = acc?;
                                acc.push(vv);
                                acc
                            }),
                        }
                    },
                );
            let bufs = bufs?;
            let writer = &mut dataset.writer;
            let writed: anyhow::Result<usize> = writer.write_all(bufs.as_slice()).await;
            if let Err(e) = writed {
                return Err(anyhow::Error::from(e));
            }
            if let Err(e) = writer.flush().await {
                return Err(anyhow::Error::from(e));
            }
            Ok(())
        }
    }

    async fn write_tokinizer_encode(
        dataset: &Arc<RwLock<Self>>, // 目测，不清楚tokinizer是否是线程安全的
        records: Vec<Record>,
        chunk_max_size: usize,
    ) -> anyhow::Result<()> {
        let s = stream::iter(records)
            .then(|r| Self::encode_to_bytes(dataset, r))
            .inspect_err(|e| error!("tokinizer encode error: {:?}", e))
            .ready_chunks(chunk_max_size)
            .then(|r| Self::write_bytes_to_dataset(dataset, r))
            .inspect_err(|e| error!("tokinizer write error: {:?}", e))
            .take_while(|r| std::future::ready(r.is_ok())); // 有错误就停止
        s.for_each_concurrent(num_cpus::get() * 3, |_| async {})
            .await;
        Ok(())
        // warn!("Dataset.bin is written size: {}", dataset.size_of_written.load(std::sync::atomic::Ordering::SeqCst));
    }
}
