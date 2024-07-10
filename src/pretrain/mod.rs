use std::{
    io,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

use anyhow::Error;

use futures::{
    sink,
    stream::{self, Stream},
    Sink, StreamExt,
};


use log::{error, info};
use serde::{Deserialize, Serialize};
use tokio::task::{self};
use tokio_stream::StreamExt as TokioStreamExt;

use crate::prelude::args;

static IMPORTED_LINES_TOTAL: AtomicU64 = AtomicU64::new(0);

#[inline(never)]
pub fn get_imported_lines() -> u64 {
    IMPORTED_LINES_TOTAL.load(Ordering::Relaxed)
}

pub type PretrainCSVFile = tokio::fs::File;

#[derive(Debug)]
enum CSVLine<L> {
    Line(L),
    /// None means EOF of file
    None,
    Error(Error),
}

impl<L> Clone for CSVLine<L> 
    where L: Clone
{
    fn clone(&self) -> Self {
        match self {
            Self::Line(l) => Self::Line(l.clone()),
            Self::None => Self::None,
            Self::Error(e) => Self::Error(anyhow::anyhow!("Error: {}", e)),
        }
    }
}

impl<L> CSVLine<L>
where
    L: Sync + Send + Unpin + Sized + 'static,
{
    fn map<F, T>(self, f: F) -> CSVLine<T>
    where
        F: FnOnce(L) -> T,
        T: Sync + Send + Unpin + Sized + 'static,
    {
        match self {
            CSVLine::Line(line) => CSVLine::Line(f(line)),
            CSVLine::None => CSVLine::None,
            CSVLine::Error(e) => CSVLine::Error(e),
        }
    }

    fn is_line(&self) -> bool {
        match self {
            CSVLine::Line(_) => true,
            _ => false,
        }
    }

    fn not_line<T>(&self) -> CSVLine<T>
    where
        T: Sync + Send + Unpin + Sized + 'static,
    {
        match self {
            CSVLine::Line(_) => unreachable!(),
            CSVLine::Error(e) => CSVLine::<T>::Error(anyhow::anyhow!("Error: {}", e)),
            CSVLine::None => CSVLine::<T>::None,
        }
    }

    fn line(&self) -> Option<&L> {
        match self {
            CSVLine::Line(ref l) => Some(l),
            _ => None,
        }
    }
}

impl CSVLine<csv::Result<Record>> {
    fn flatten(self) -> CSVLine<Record> {
        match self {
            CSVLine::Line(Ok(r)) => CSVLine::Line(r),
            CSVLine::Line(Err(e)) => CSVLine::Error(e.into()),
            CSVLine::Error(e) => CSVLine::Error(e),
            CSVLine::None => CSVLine::None,
        }
    }
}

impl From<io::Result<Option<String>>> for CSVLine<String> {
    fn from(r: io::Result<Option<String>>) -> Self {
        match r {
            Ok(Some(line)) => CSVLine::Line(line),
            Ok(None) => CSVLine::None,
            Err(e) => CSVLine::Error(e.into()),
        }
    }
}
impl From<Option<Result<String, io::Error>>> for CSVLine<String> {
    fn from(r: Option<Result<String, io::Error>>) -> Self {
        match r {
            Some(Ok(line)) => CSVLine::Line(line),
            Some(Err(e)) => CSVLine::Error(e.into()),
            None => CSVLine::None,
        }
    }
}
impl From<Result<String, io::Error>> for CSVLine<String> {
    fn from(r: Result<String, io::Error>) -> Self {
        match r {
            Ok(line) => CSVLine::Line(line),
            Err(e) => CSVLine::Error(e.into()),
        }
    }
}
impl From<csv::Result<Record>> for CSVLine<Record> {
    fn from(r: csv::Result<Record>) -> Self {
        match r {
            Ok(line) => CSVLine::Line(line),
            Err(e) => CSVLine::Error(e.into()),
        }
    }
}

trait IntoStream {
    fn into_stream(path: impl AsRef<Path>) -> impl Stream<Item = CSVLine<Record>> + Unpin;
}

// 75秒，100*20 0000
// #[cfg(feature = "unstable")]
impl IntoStream for PretrainCSVFile {
    fn into_stream(path: impl AsRef<Path>) -> impl Stream<Item = CSVLine<Record>> + Unpin {
        let s = async_stream::stream! {
            let path = path.as_ref().to_owned();
            let opened = PretrainCSVFile::open(&path).await;
            let file = match opened {
                Ok(file) => file,
                Err(e) => {
                    error!("Error opening file({}): {}", path.display(), e);
                    yield vec![CSVLine::Error(e.into())];
                    return;
                }
            };

            let current_thread = std::thread::current();
            let current_thread_id = current_thread.id();
            let current_thread_name = current_thread.name().unwrap_or("unnamed");
            log::debug!("{current_thread_id:?}/{current_thread_name} opened file({})", path.display());

            let buf_reader = std::io::BufReader::new(file.into_std().await);
            let mut rdr = csv::ReaderBuilder::new().delimiter(args().csv_delimiter as u8).from_reader(buf_reader);

            const CAP: usize = 100;
            let mut vec = Vec::<CSVLine<Record>>::with_capacity(CAP + 3);
            for result in rdr.deserialize() {
                vec.push(CSVLine::from(result));
                //.CSVLine::from(result);
                if vec.len() >= CAP {
                    yield vec;
                    // tokio::task::yield_now().await;
                    vec = Vec::<CSVLine<Record>>::with_capacity(CAP + 3);
                }
            }
            if vec.len() > 0 {
                yield vec;
            }
            
            yield vec![CSVLine::None];
            // info!("{current_thread_id:?}/{current_thread_name} import file({}) done, all-imported {}. ",
            //     path.display(), get_imported_lines());
        };
        Box::pin(s.flat_map(stream::iter))
    }
}

// 190秒，100*20 0000
#[cfg(feature = "unstable")]
impl IntoStream for PretrainCSVFile {
    fn into_stream(path: impl AsRef<Path>) -> impl Stream<Item = CSVLine<Record>> + Unpin {
        use csv::StringRecord;
        use async_stream::stream;
        use tokio::io::AsyncBufReadExt;
        use log::debug;

        Box::pin(stream! {
            let path = path.as_ref().to_owned();
            let opened = PretrainCSVFile::open(&path).await;
            let file = match opened {
                Ok(file) => file,
                Err(e) => {
                    error!("Error opening file({}): {}", path.display(), e);
                    yield CSVLine::Error(e.into());
                    return;
                }
            };

            let current_thread = std::thread::current();
            let current_thread_id = current_thread.id();
            let current_thread_name = current_thread.name().unwrap_or("unnamed");
            debug!("{current_thread_id:?}/{current_thread_name} opened file({})", path.display());

            let buf_reader = tokio::io::BufReader::with_capacity(1 * 1024 * 1024, file);
            let mut lines = buf_reader.lines();
            let header = lines.next_line().await;
            let header = CSVLine::from(header);
            if !header.is_line() {
                yield header.not_line();
                return;
            }

            fn split_line(l: String) -> StringRecord {
                let v: Vec<&str> = l.split(args().csv_delimiter).collect();
                let mut sr = StringRecord::from(v);
                sr.trim();
                sr
            }
            let header: CSVLine<StringRecord> = header.map(&split_line);
            let header = CSVLine::<StringRecord>::line(&header);

            loop {
                let line = lines.next_line().await;
                let line = CSVLine::from(line);
                if !line.is_line() {
                    yield line.not_line();
                    break;
                }

                let line = CSVLine::map(line, &split_line);
                let line = CSVLine::map(line, |sr| StringRecord::deserialize::<'_, Record>(&sr, header));
                yield line.flatten();
            }

            info!("{current_thread_id:?}/{current_thread_name} import file({}) done, all-imported {}. ",
                path.display(), get_imported_lines());
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Record {
    #[serde(deserialize_with = "csv::invalid_option")]
    id: Option<u64>,
    comment: String,
    sentiment: u8,
}

impl Record {
    pub fn new(id: Option<u64>, comment: String, sentiment: u8) -> Self {
        Self {
            id,
            comment,
            sentiment,
        }
    }

    pub fn comment(&self) -> &str {
        &self.comment
    }

    pub fn sentiment(&self) -> u8 {
        self.sentiment
    }

    pub fn id(&self) -> Option<u64> {
        self.id
    }
}

pub fn records_of_train_files() -> impl Stream<Item = Record> + Unpin {
    let s = stream::iter(args().pretrained_files.iter().flat_map(|&pb| pb));
    // let s = StreamExt::flat_map_unordered(s, Some(args().unordered_files), |f| {
    let s = StreamExt::flat_map(s, |f| {
        Box::pin(StreamExt::filter_map(
            PretrainCSVFile::into_stream(f),
            |f| async move {
                match f {
                    CSVLine::Line(r) => {
                        IMPORTED_LINES_TOTAL.fetch_add(1, Ordering::Relaxed);
                        Some(r)
                    }
                    CSVLine::Error(e) => {
                        error!("Error reading file: {}", e);
                        None
                    }
                    CSVLine::None => None,
                }
            },
        ))
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

pub fn pretrain_sink() -> impl Sink<Vec<Record>, Error = anyhow::Error> + Unpin {
    let sink = sink::unfold(0, |mut sum, records: Vec<Record>| async move {
        let t = task::spawn(async move {
            sum += records.len();
            // info!("records: {:?}, pretrain items: {}", &records, sum);

            // let current_thread = std::thread::current();
            // let current_thread_id = current_thread.id();
            // let current_thread_name = current_thread.name().unwrap_or("unnamed");
            // info!(
            //     "pretrain items: {} / imported total: {}, current thread = {current_thread_id:?}/{current_thread_name}",
            //     sum,
            //     get_imported_lines()
            // );
            Ok::<usize, anyhow::Error>(sum)
        })
        .await??;
        Ok(t)
    });
    Box::pin(sink)
}

pub async fn pretrain_do() -> anyhow::Result<()> {
    let trained = StreamExt::map(chunks_timeout_of_train_files(), Ok::<_, anyhow::Error>)
        .forward(pretrain_sink())
        .await;

    if let Err(ref e) = trained {
        error!("Error training: {}", e);
    }

    trained
}

pub async fn spawn_pretrain_task() -> anyhow::Result<()> {
    let pretrain_begin = Instant::now();
    let r = task::spawn(pretrain_do()).await?;
    info!(
        "Pretrain cost: {:?}, cpu cores = {}",
        pretrain_begin.elapsed(),
        num_cpus::get()
    );
    r
}
