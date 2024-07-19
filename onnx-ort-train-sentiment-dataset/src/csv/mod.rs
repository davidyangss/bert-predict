use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};


use futures::{
    sink,
    stream::{self, Stream},
    Sink, StreamExt,
};

use tokio::task::{self};
use tokio_stream::StreamExt as TokioStreamExt;

mod record;
use record::Record;
use tracing::Level;

use super::prelude::*;

static IMPORTED_LINES_TOTAL: AtomicU64 = AtomicU64::new(0);
#[inline(never)]
pub fn get_imported_lines() -> u64 {
    IMPORTED_LINES_TOTAL.load(Ordering::Relaxed)
}

pub type TrainCSV = tokio::fs::File;

trait IntoStream {
    fn into_stream(path: impl AsRef<Path>) -> impl Stream<Item = anyhow::Result<Record>> + Unpin;
}

impl IntoStream for TrainCSV {
    fn into_stream(path: impl AsRef<Path>) -> impl Stream<Item = anyhow::Result<Record>> + Unpin {
        let s = async_stream::stream! {
            let import_begin = Instant::now();
            let path = path.as_ref().to_owned();
            let rdr = csv_async::AsyncReaderBuilder::new()
                .delimiter(args().csv_delimiter as u8)
                .create_deserializer(TrainCSV::open(&path).await?);
            let mut records = rdr.into_deserialize::<Record>();
            while let Some(record) = StreamExt::next(&mut records).await {
                let r = match record {
                    Ok(r) => anyhow::Result::Ok(r),
                    Err(e) => anyhow::Result::Err(e.into()),
                };
                yield r
            }
            info!("Done. import csv: {} {}", import_begin.elapsed().as_secs_f32(), path.display());
        };
        Box::pin(s)
    }
}

pub fn records_of_train_files() -> impl Stream<Item = Record> + Unpin {
    let s = stream::iter(args().csvs.iter());
    // let s = StreamExt::flat_map_unordered(s, Some(args().unordered_files), |f| {
    let s = StreamExt::flat_map(s, |f| {
        Box::pin(StreamExt::filter_map(TrainCSV::into_stream(f), |r| async {
            match r {
                Ok(r) => {
                    IMPORTED_LINES_TOTAL.fetch_add(1, Ordering::Relaxed);
                    Some(r)
                }
                Err(e) => {
                    error!("Error reading csv({}): {:?}", f.display(), e);
                    None
                }
            }
        }))
    });
    Box::pin(s)
}

pub fn chunks_timeout_of_train_files() -> impl Stream<Item = Vec<Record>> {
    // records_of_train_files().ready_chunks(args().chunk_max_size)
    TokioStreamExt::chunks_timeout(
        records_of_train_files(),
        args().chunk_max_size,
        Duration::from_millis(args().chunk_timeout),
    )
}

pub fn dataset_sink() -> impl Sink<Vec<Record>, Error = anyhow::Error> + Unpin {
    let sink = sink::unfold(0, |mut sum, records: Vec<Record>| async move {
        let t = task::spawn(async move {
            sum += records.len();
            // info!("records: {:?}, dataset items: {}", &records, sum);
            info!("dataset items: {} / {}", IMPORTED_LINES_TOTAL.load(Ordering::Relaxed), sum);

            Ok::<usize, anyhow::Error>(sum)
        })
        .await??;
        Ok(t)
    });
    Box::pin(sink)
}

pub async fn dataset_do() -> anyhow::Result<()> {
    let r = StreamExt::map(chunks_timeout_of_train_files(), Ok::<_, anyhow::Error>)
        .forward(dataset_sink())
        .await;

    if let Err(ref e) = r {
        error!("Error convert to dataset: {}", e);
    }

    r
}

#[tracing_attributes::instrument(level = Level::INFO, name = "spawn_dataset_task")]
pub async fn spawn_dataset_task() -> anyhow::Result<()> {
    let dataset_begin = Instant::now();
    let r = task::spawn(dataset_do()).await?;
    info!("dataset cost: {:?},", dataset_begin.elapsed());
    r
}
