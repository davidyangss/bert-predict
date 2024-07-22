use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use futures::{
    stream::{self, Stream},
    StreamExt,
};

mod record;
pub use record::Record;
use tracing::{event, field, info_span, Level};

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
            let path = path.as_ref();
            let import_span = info_span!("import_csv", path = field::Empty);
            // let _guard = import_span.record("path", path.file_name().map(|p| p.to_str()).unwrap()).enter();
            let _guard = import_span.enter();
            let import_b = Instant::now();
            let rdr = csv_async::AsyncReaderBuilder::new()
                .delimiter(args().csv_delimiter as u8)
                .create_deserializer(TrainCSV::open(path).await?);
            event!(Level::INFO, "Opened {}", path.display());
            let mut records = rdr.into_deserialize::<Record>();
            while let Some(record) = StreamExt::next(&mut records).await {
                match record {
                    Ok(r) => yield anyhow::Result::Ok(r),
                    Err(e) => {
                        yield anyhow::Result::Err(e.into());
                        break;
                    }
                };
            }
            event!(Level::INFO, "Done. cost {} / {}", import_b.elapsed().as_millis(), path.display());
        };
        Box::pin(s)
    }
}

pub fn records_of_train_files() -> impl Stream<Item = Record> + Unpin {
    let s = stream::iter(args().csvs.iter());
    let s = StreamExt::flat_map_unordered(s, args().unordered, |f| {
        // let s = StreamExt::flat_map(s, |f| {
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
    records_of_train_files().ready_chunks(args().chunk_max_size)
    // TokioStreamExt::chunks_timeout(
    //     records_of_train_files(),
    //     args().chunk_max_size,
    //     Duration::from_millis(args().chunk_timeout),
    // )
}
