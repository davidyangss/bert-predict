use std::{
    path::{Path, PathBuf},
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
pub type TrainCSV = tokio::fs::File;

trait IntoStream {
    fn into_stream(
        path: impl AsRef<Path>,
        csv_delimiter: char,
    ) -> impl Stream<Item = anyhow::Result<Record>> + Unpin;
}

#[cfg(feature = "tokio-async-stream")]
impl IntoStream for TrainCSV {
    fn into_stream(
        path: impl AsRef<Path>,
        csv_delimiter: char,
    ) -> impl Stream<Item = anyhow::Result<Record>> + Unpin {
        // use futures::TryFutureExt;

        let s = async_stream::stream! {
            let path = path.as_ref();
            let import_span = info_span!("import_csv", path = field::Empty);
            // let _guard = import_span.record("path", path.file_name().map(|p| p.to_str()).unwrap()).enter();
            let _guard = import_span.enter();
            let import_b = Instant::now();
            let rdr = csv_async::AsyncReaderBuilder::new()
                .delimiter(csv_delimiter as u8)
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

#[cfg(feature = "futures-async-stream")]
impl IntoStream for TrainCSV {
    fn into_stream(path: impl AsRef<Path>) -> impl Stream<Item = anyhow::Result<Record>> + Unpin {
        use std::path::PathBuf;

        #[futures_async_stream::try_stream(boxed, ok = Record, error = anyhow::Error)]
        async fn line_stream(path: PathBuf) {
            let import_span = info_span!("import_csv", path = field::Empty);
            let _guard = import_span.enter();
            let import_b = Instant::now();

            let file = TrainCSV::open(&path).await?;
            let rdr = csv_async::AsyncReaderBuilder::new()
                .delimiter(args().csv_delimiter as u8)
                .create_deserializer(file);
            event!(Level::INFO, "Opened {}", &path.display());
            let mut records = rdr.into_deserialize::<Record>();
            while let Some(record) = StreamExt::next(&mut records).await {
                yield record?;
            }
            event!(
                Level::INFO,
                "Done. cost {} / {}",
                import_b.elapsed().as_millis(),
                path.display()
            );
        }
        let s = line_stream(path.as_ref().to_path_buf());
        s
    }
}

pub fn train_records<'a>(
    files: &'a [PathBuf],
    csv_delimiter: char,
) -> impl Stream<Item = Record> + Unpin + 'a {
    let s = stream::iter(files);
    // let s = StreamExt::flat_map_unordered(s, args().unordered, |f| {
    let s = StreamExt::flat_map(s, move |f| {
        Box::pin(StreamExt::filter_map(
            TrainCSV::into_stream(f, csv_delimiter),
            |r| async {
                match r {
                    Ok(r) => Some(r),
                    Err(e) => {
                        error!("Error reading csv({}): {:?}", f.display(), e);
                        None
                    }
                }
            },
        ))
    });
    Box::pin(s)
}

pub fn chunks_train_records<'a>(
    files: &'a [PathBuf],
    csv_delimiter: char,
    chunk_max_size: usize,
) -> impl Stream<Item = Vec<Record>> + Unpin + 'a {
    train_records(files, csv_delimiter).ready_chunks(chunk_max_size)
}
