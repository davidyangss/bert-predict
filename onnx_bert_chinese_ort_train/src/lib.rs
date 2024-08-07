use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Instant,
};

use anyhow::Ok;
use futures::{stream, FutureExt, StreamExt, TryStreamExt};
use lazy_static::lazy_static;
use onnx_bert_chinese_ort_train_dataset::text_label::{DatasetReader, TextLabel};
use regex::Regex;
use tokio::{
    fs::{File, OpenOptions},
    sync::mpsc,
    task::JoinSet,
};
use tracing::{error, info, Level};

pub mod ort_training;

// format!("dataset-{id}-{total}-{ids_max_len}.bin"));
fn exact_lines_maxlen_from_filename(file_name: &str) -> (usize, usize) {
    lazy_static! {
        static ref RE: Regex =
            Regex::new(r"^dataset-(?P<id>\d+)-(?P<lines>\d+)-(?P<ids_max_len>\d+)").unwrap();
    }
    let r = RE.captures(file_name).and_then(|cap| {
        let lines = cap.name("lines").map(|lines: regex::Match| lines.as_str());
        let ids_max_len = cap
            .name("ids_max_len")
            .map(|ids_max_len: regex::Match| ids_max_len.as_str());
        let lines = lines
            .expect(&format!("Can not got lines from {file_name}"))
            .parse::<usize>()
            .expect(&format!("can not parse lines{lines:?} to usize"));
        let ids_max_len = ids_max_len
            .expect(&format!("Can not got ids_max_len from {file_name}"))
            .parse::<usize>()
            .expect(&format!(
                "can not parse ids_max_len{ids_max_len:?} to usize"
            ));
        Some((lines, ids_max_len))
    });
    r.unwrap()
}

#[test]
fn test_exact_lines_maxlen_from_filename() {
    let (lines, ids_max_len) = exact_lines_maxlen_from_filename("dataset-0-100-10.bin");
    assert_eq!(lines, 100);
    assert_eq!(ids_max_len, 10);
}

fn exact_lines_maxlen(files: &[PathBuf]) -> anyhow::Result<(usize, usize)> {
    let total_lines = files
        .iter()
        .map(|f| exact_lines_maxlen_from_filename(f.file_name().unwrap().to_str().unwrap()))
        .fold(Ok((0, 0)), |sum, r| {
            let sum = sum?;
            let (l, m) = r;
            let (sum_l, sum_m) = sum;
            Ok((sum_l + l, sum_m.max(m)))
        });
    total_lines
}

#[derive(Debug, Clone)]
pub struct NextTraining {
    total_records: OnceLock<usize>,
    ids_max_len: OnceLock<usize>,
    dataset_bin: Vec<PathBuf>,
    bin_file_chunk_size: Option<usize>,
    training_batch_size: usize,
    tx: mpsc::Sender<Vec<TextLabel>>,
}

impl NextTraining {
    pub fn new(
        dataset_bin: Vec<PathBuf>,
        bin_file_chunk_size: Option<usize>,
        training_batch_size: usize,
        tx: mpsc::Sender<Vec<TextLabel>>,
    ) -> anyhow::Result<Self> {
        let (lines, ids_max_len) = exact_lines_maxlen(&dataset_bin)?;
        Ok(Self {
            total_records: OnceLock::from(lines),
            ids_max_len: OnceLock::from(ids_max_len),
            dataset_bin,
            bin_file_chunk_size,
            training_batch_size,
            tx,
        })
    }
    pub fn total_records(&self) -> usize {
        self.total_records.get().unwrap().clone()
    }

    pub fn ids_max_len(&self) -> usize {
        self.ids_max_len.get().unwrap().clone()
    }
}

impl NextTraining {
    pub fn update_dataset_bin(&self, dataset_bin: Vec<PathBuf>) -> Self {
        let mut c = self.clone();
        c.dataset_bin = dataset_bin;
        c
    }

    #[tracing_attributes::instrument(level = Level::INFO, name = "training_task", skip(self))]
    pub async fn next_training(self: Self) -> anyhow::Result<()> {
        let this = Arc::new(self);
        let dataset_begin = Instant::now();

        let t = this.clone();
        let trained = tokio::spawn(t.do_next_training()).await??;

        let report = format!(
            "Next training cost: {:?}, trained: {trained}, total: {}, ids_max_len: {}",
            dataset_begin.elapsed(),
            this.total_records(),
            this.ids_max_len()
        );
        if trained != this.total_records() {
            error!("{report}");
        } else {
            info!("{report}");
        }
        Ok(())
    }

    async fn do_next_training(self: Arc<Self>) -> anyhow::Result<usize> {
        info!(
            "Next training do, will import {} records by filename",
            self.total_records()
        );

        if self.bin_file_chunk_size.is_none() || Some(0) == self.bin_file_chunk_size {
            let s = self.next_training_by_task(0).await?;
            return Ok(s);
        }

        let mut tasks = self
            .dataset_bin
            .chunks(
                self.bin_file_chunk_size
                    .ok_or(anyhow::anyhow!("bin_file_chunk_size error"))?,
            )
            .enumerate()
            .map(|(id, files)| {
                let n = Arc::new(self.update_dataset_bin(files.to_vec()));
                n.next_training_by_task(id)
            })
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
                Some(Result::Ok(Err(e))) => {
                    error!("next training, continue. error: {:?}", e);
                    continue;
                }
                Some(Err(e)) => {
                    error!("next training, break. error: {:?}", e);
                    break;
                }
            }
        }
        Ok(sum)
    }

    // 一小批文件，对应 一个dataset_sink_writer
    #[tracing_attributes::instrument(level = Level::INFO, name = "Next_training", skip(self))]
    async fn next_training_by_task(self: Arc<Self>, id: usize) -> anyhow::Result<usize> {
        // 坑啊，如果直接将其直接写入map出，将被推断为早期绑定，方法参数生命周期参与了类型定义。而单独写在此处，HRBT。
        let clone_pathbuf = |p: &PathBuf| p.clone();
        let dataset_begin = Instant::now();
        info!("Begin. training: id={}, files={:?}", id, self.dataset_bin);
        let s = stream::iter(self.dataset_bin.iter())
            .map(clone_pathbuf) //.map(|p: &PathBuf| p.clone()) 这样就嘎了
            .then(|f| async move {
                let f = OpenOptions::new().read(true).open(f).await?;
                let r = DatasetReader::<TextLabel, File>::new(f).await?;
                Ok(r)
            })
            .map_ok(|read| read.into_stream())
            .try_flatten()
            .try_ready_chunks(self.training_batch_size)
            .map_err(|e| e.into())
            .and_then(|chunk| {
                let sender = self.tx.clone();
                async move {
                    let chunk_size = chunk.len();
                    sender.send(chunk).await?;
                    Ok(chunk_size)
                }
            })
            .inspect_err(|e| error!("Do training by files, id = {id}: error = {e:?}"))
            .take_while(|r| futures::future::ready(r.is_ok()))
            .try_fold(0_usize, |sum, trained| async move { Ok(sum + trained) })
            .inspect(|rr| {
                info!(
                    "Done. training by files, id = {id}: trained = {rr:?}, cost = {}",
                    dataset_begin.elapsed().as_secs_f32()
                );
            });
        s.await
    }
}
