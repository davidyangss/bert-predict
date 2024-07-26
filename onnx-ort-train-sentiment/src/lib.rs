use std::{
    cell::OnceCell,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Instant,
};

use anyhow::Ok;
use futures::{stream, StreamExt, TryStreamExt};
use onnx_ort_train_sentiment_dataset::text_label::{DatasetReader, TextLabel};
use ort_training::OrtTraining;
use tokio::{
    fs::{File, OpenOptions},
    task::JoinSet,
};
use tracing::{error, info, Level};
use tracing_subscriber::registry::Data;

pub mod ort_training;

fn total_lines(files: &[PathBuf]) -> anyhow::Result<usize> {
    let total_lines = files
        .iter()
        .map(|f| {
            let n = f.file_name().unwrap().to_str().unwrap();
            let n = n
                .split('-')
                .last()
                .ok_or(anyhow::anyhow!("file name split by - error: {n}"))?
                .split('.')
                .next()
                .ok_or(anyhow::anyhow!("file name split by . error: {n}"))?
                .parse::<usize>()
                .map_err(|e| anyhow::anyhow!("parse usize error: {n} {e}"))?;
            Ok(n)
        })
        .fold(Ok(0), |sum, r| {
            let sum = sum?;
            match r {
                Result::Ok(n) => Ok(sum + n),
                Err(e) => Err(e),
            }
        });
    total_lines
}

pub struct Training {
    total_records: OnceCell<usize>,
    dataset_bin: Vec<PathBuf>,
    bin_chunks: Option<usize>,
    chunk_max_size: usize,
    ort_training: Arc<RwLock<OrtTraining>>,
}

impl Training {
    pub fn new(
        dataset_bin: Vec<PathBuf>,
        bin_chunks: Option<usize>,
        chunk_max_size: usize,
        ort_training: Arc<RwLock<OrtTraining>>,
    ) -> anyhow::Result<Self> {
        let total = total_lines(&dataset_bin)?;
        Ok(Self {
            total_records: OnceCell::from(total),
            dataset_bin,
            bin_chunks,
            chunk_max_size,
            ort_training,
        })
    }
    pub fn total_records(&self) -> usize {
        self.total_records.get().unwrap().clone()
    }
}

impl Training {
    #[tracing_attributes::instrument(level = Level::INFO, name = "training_task", skip(self))]
    pub async fn spawn_training_task(&self) -> anyhow::Result<()> {
        let dataset_begin = Instant::now();
        let trained = tokio::spawn(self.do_training()).await??;
        info!(
            "train cost: {:?}, trained: {trained}, total: {}",
            dataset_begin.elapsed(),
            self.total_records()
        );
        Ok(())
    }

    async fn do_training(&self) -> anyhow::Result<usize> {
        info!("training total lines: {}", self.total_records());

        if self.bin_chunks.is_none() || Some(0) == self.bin_chunks {
            let s = self
                .inspect_do_by_files(0, self.dataset_bin.to_vec())
                .await?;
            return Ok(s);
        }

        let mut tasks = self
            .dataset_bin
            .chunks(self.bin_chunks.ok_or(anyhow::anyhow!("bin_chunks error"))?)
            .enumerate()
            .map(|(id, files)| self.inspect_do_by_files(id, files.to_vec()))
            .fold(JoinSet::new(), |mut set, fut| {
                set.spawn(fut);
                set
            });

        let mut sum = 0_usize;
        loop {
            let r = tasks.join_next().await;
            match r {
                None => break,
                Some(Result::Ok(Result::Ok(trained))) => sum += trained,
                Some(Err(e)) => {
                    error!("dataset_do: {:?}", e);
                    break;
                }
                Some(Result::Ok(Err(e))) => {
                    error!("dataset_do: {:?}", e);
                    continue;
                }
            }
        }
        Ok(sum)
    }

    // 一小批文件，对应 一个dataset_sink_writer
    #[tracing_attributes::instrument(level = Level::INFO, name = "inspect_training", skip(files, self))]
    async fn inspect_do_by_files(&self, id: usize, files: Vec<PathBuf>) -> anyhow::Result<usize> {
        let r = self.do_training_by_files(id, &files).await;
        match r {
            Result::Ok(t) => {
                info!("inspect_do_by_files: id={}, total = {}", id, t);
                Ok(t)
            }
            Err(e) => {
                error!("inspect_do_by_files: id={}, error = {:?}", id, e);
                Err(e)
            }
        }
    }
    async fn do_training_by_files(&self, id: usize, files: &[PathBuf]) -> anyhow::Result<usize> {
        let dataset_begin = Instant::now();
        info!("Begin. training: id={}, files={:?}", id, files);
        let s = stream::iter(files)
            .then(|f| async move {
                let f = OpenOptions::new().read(true).open(&f).await?;
                let r = DatasetReader::<TextLabel, File>::new(f).await?;
                Ok(r)
            })
            .map_ok(|read| read.into_stream())
            .try_flatten()
            .inspect_err(|e| error!("Do training by files, id = {id}: error = {:?}", e))
            .try_ready_chunks(self.chunk_max_size)
            .map_ok(|chunk| {
                // let mut ort_training = self.ort_training.write()?;
                // let loss = ort_training.setup(chunk);

                Ok(chunk.len())
            });

        todo!()
    }
}
