// #![feature(coroutines, proc_macro_hygiene, stmt_expr_attributes)]
// #![feature(cursor_remaining)]

pub mod csv;
pub mod text_label;

use std::{
    fs::OpenOptions,
    future::Future,
    path::{Path, PathBuf},
    sync::{atomic::AtomicUsize, Arc},
};

use anyhow::{anyhow, Ok};
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
use text_label::{DatasetWriter, TextLabel, TextLabelBytes};
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
        let tokinizer = Tokenizer::from_file(tokenizer_json.clone())
            .map_err(|_| anyhow!("create tokenizer fail"))?;
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
    pub fn ids_max_len(&self) -> usize {
        self.writer.ids_max_len()
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
        training_batch_size: usize,
    ) -> impl Sink<Vec<Record>, Error = anyhow::Error> + Unpin {
        let sink = sink::unfold(dataset, move |s, records| async move {
            Self::write_tokinizer_encode(&s.clone(), records, training_batch_size).await?;
            Ok(s)
        });
        Box::pin(sink)
    }

    async fn write_tokinizer_encode(
        dataset: &Arc<RwLock<Self>>, // 目测，不清楚tokinizer是否是线程安全的
        records: Vec<Record>,
        training_batch_size: usize,
    ) -> anyhow::Result<()> {
        let s = stream::iter(records)
            .then(|r| Self::encode_to_bytes(dataset, r))
            .inspect_err(|e| error!("tokinizer encode error: {:?}", e))
            .try_ready_chunks(training_batch_size)
            .inspect(|r| trace!("tokinizer encode to bytes, chunk size: {:?}", r))
            .map_err(|e| anyhow::Error::from(e))
            .take_while(|r| futures::future::ready(r.is_ok())) // 有错误就停止
            .and_then(|vec| Self::write_bytes_to_dataset(dataset, vec))
            .inspect_err(|e| error!("tokinizer write error: {:?}", e))
            .take_while(|r| std::future::ready(r.is_ok())); // 有错误就停止
        s.for_each_concurrent(num_cpus::get() * 3, |_| async {})
            .await;
        Ok(())
        // warn!("Dataset.bin is written size: {}", dataset.size_of_written.load(std::sync::atomic::Ordering::SeqCst));
    }

    fn encode_to_bytes(
        dataset: &Arc<RwLock<Self>>,
        r: Record,
    ) -> impl Future<Output = anyhow::Result<TextLabel>> + 'static {
        let dataset = dataset.clone();
        async move {
            let dataset = dataset.read().await;
            dataset.lines_plus_plus();
            let ids = dataset
                .tokinizer
                .encode(r.comment(), true)
                .map_err(|e| {
                    anyhow::anyhow!("tokinizer encode to bytes, error:{}", e)
                })?;
            let ids = ids.get_ids();
            let id_bytes = dataset.writer.item_style().tokenizer_ids_as_bytes(ids);
            trace!(
                "tokinizer comment <{}> to {:X} / [{:?}]",
                r.comment(),
                id_bytes.len(),
                hex::encode(&id_bytes)
            );
            // let label_bytes = dataset.writer.item_style().tokenizer_ids_as_bytes(
            //     &dataset
            //         .tokinizer
            //         .encode(r.sentiment().to_string(), false)
            //         .map_err(|e| {
            //             anyhow::anyhow!("tokinizer encode to bytes, error:{}", e.to_string())
            //         })?
            //         .get_ids(),
            // );
            let label_bytes = r.sentiment().to_le_bytes();
            let label_bytes = label_bytes.map(|i| i as u32);
            let label_bytes = dataset
                .writer
                .item_style()
                .tokenizer_ids_as_bytes(&label_bytes);
            trace!(
                "tokinizer sentiment <{}> to {} / [{:?}]",
                r.sentiment(),
                label_bytes.len(),
                hex::encode(&label_bytes)
            );
            Ok(dataset.writer.style_bytes(id_bytes, label_bytes.to_vec()))
        }
    }

    fn write_bytes_to_dataset(
        dataset: &Arc<RwLock<Self>>,
        bufs: Vec<TextLabel>,
    ) -> impl Future<Output = anyhow::Result<()>> + 'static {
        let dataset = dataset.clone();
        async move {
            // 写文件
            let mut dataset = dataset.write().await;
            let writer = &mut dataset.writer;
            let writed: anyhow::Result<usize> = writer.write_all(bufs.as_slice()).await;
            if let Err(e) = writed {
                return Err(anyhow::Error::from(e));
            }
            if let Err(e) = writer.flush().await {
                return Err(anyhow::Error::from(e));
            }
            let ids_max_len = bufs
                .iter()
                .map(|tx_lb| tx_lb.tokenizer_ids_len(tx_lb.id_bytes()))
                .max();
            writer.update(ids_max_len);
            Ok(())
        }
    }
}
