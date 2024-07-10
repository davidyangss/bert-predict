use std::{io::BufWriter, path::PathBuf, time::Instant};

use anyhow::Ok;
use bert::pretrain::Record;
use futures::{stream, StreamExt};
use tokio::fs::{self, remove_file, File};

// cargo run --example gen-csvs --features="pretrain"
#[tokio::main(flavor = "multi_thread", worker_threads = 10)]
async fn main() -> anyhow::Result<()> {
    let begin = Instant::now();
    tokio::spawn(gentask()).await?;
    // gentask().await;
    println!("gen csvs cost: {:?}", begin.elapsed());
    Ok(())
}
async fn gentask() {
    let root = &PathBuf::from("./target/gen-csvs");
    // not exist path, create it
    if !root.exists() {
        let _ = fs::create_dir_all(&root).await;
    }

    const FILES: u64 = 100;
    const SIZE_OF_FILE: u64 = 200000;
    const FLUSH_CAPACITY_OF_WRITE_ONE_FILE: u64 = 100;
    const CONCURRENT_FILES: u64 = 20;

    let gener = Box::pin(stream::iter(0..FILES))
    .flat_map_unordered(CONCURRENT_FILES as usize, |f| {
        // let thread_id1 = thread::current().id();
        // let thread_name1 = thread::current().name().unwrap_or("unnamed_1").to_owned();
        let ws = async_stream::stream! {
            let path = &root.join(format!("text-{}.csv", f));
            if path.exists() {
                remove_file(&path).await?;
            }

            yield {
                let csv = File::create(path).await?;
                let mut wtr = csv::Writer::from_writer(BufWriter::new(csv.into_std().await));

                // let current_thread_id = thread::current().id();
                // let current_thread_name = thread::current().name().unwrap_or("unnamed_1").to_owned();
                // println!("write file: {}, current thread = {current_thread_id:?}/{current_thread_name} < {thread_id1:?}/{thread_name1}", path.display());

                for i in 0..SIZE_OF_FILE {
                    let s = "涉及党内有争议尚未做出结论的重大问题以及重大政治历史事件不宜公开的档案、材料；涉及地方重大事件不宜公开的档案、材料";
                    wtr.serialize(Record::new(Some(f * SIZE_OF_FILE + i), s.to_owned(), 1))?;
                    if i % FLUSH_CAPACITY_OF_WRITE_ONE_FILE == 0 {
                        wtr.flush()?;
                    }
                }
                wtr.flush()?;

                print!(" --pretrain-file=\"{}\"", path.display());

                Ok(())
            };
        };
        Box::pin(ws)
    })
    .for_each_concurrent(CONCURRENT_FILES as usize, |r| async move{
        // let current_thread_id = thread::current().id();
        // let current_thread_name = thread::current().name().unwrap_or("unnamed_1").to_owned();
        // println!("for_each_concurrent:, current thread = {current_thread_id:?}/{current_thread_name}");

        match r {
            Result::Ok(_) => {},
            Err(e) => {
                println!("error: {:?}", e);
            }
        }
    });

    gener.await
}
