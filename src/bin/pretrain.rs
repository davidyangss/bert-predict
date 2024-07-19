use std::{
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

use bert::prelude::*;
use log::info;
use tokio::runtime::{self, HistogramScale};

// cargo run --bin pretrain --features="pretrain" -- --model-config="/home/yangss/Downloads/config.json" --model-vocab="/home/yangss/Downloads/vocab.txt" --model-ot="/home/yangss/Downloads/rust_model.ot" --pretrain-file="/home/yangss/Downloads/comments.csv" --pretrain-file="/home/yangss/Downloads/comments1.csv"
// cargo run --bin pretrain --features="pretrain" -- --unordered-files=13 --model-config="/home/yangss/Downloads/config.json" --model-vocab="/home/yangss/Downloads/vocab.txt" --model-ot="/home/yangss/Downloads/rust_model.ot" --pretrain-file="./target/gen-csvs/text-4.csv" --pretrain-file="./target/gen-csvs/text-15.csv" --pretrain-file="./target/gen-csvs/text-0.csv" --pretrain-file="./target/gen-csvs/text-1.csv" --pretrain-file="./target/gen-csvs/text-2.csv" --pretrain-file="./target/gen-csvs/text-6.csv" --pretrain-file="./target/gen-csvs/text-10.csv" --pretrain-file="./target/gen-csvs/text-16.csv" --pretrain-file="./target/gen-csvs/text-17.csv" --pretrain-file="./target/gen-csvs/text-18.csv" --pretrain-file="./target/gen-csvs/text-19.csv" --pretrain-file="./target/gen-csvs/text-9.csv" --pretrain-file="./target/gen-csvs/text-3.csv" --pretrain-file="./target/gen-csvs/text-5.csv" --pretrain-file="./target/gen-csvs/text-7.csv" --pretrain-file="./target/gen-csvs/text-8.csv" --pretrain-file="./target/gen-csvs/text-11.csv" --pretrain-file="./target/gen-csvs/text-12.csv" --pretrain-file="./target/gen-csvs/text-13.csv" --pretrain-file="./target/gen-csvs/text-14.csv" --pretrain-file="./target/gen-csvs/text-23.csv" --pretrain-file="./target/gen-csvs/text-30.csv" --pretrain-file="./target/gen-csvs/text-20.csv" --pretrain-file="./target/gen-csvs/text-21.csv" --pretrain-file="./target/gen-csvs/text-22.csv" --pretrain-file="./target/gen-csvs/text-25.csv" --pretrain-file="./target/gen-csvs/text-24.csv" --pretrain-file="./target/gen-csvs/text-29.csv" --pretrain-file="./target/gen-csvs/text-31.csv" --pretrain-file="./target/gen-csvs/text-37.csv" --pretrain-file="./target/gen-csvs/text-33.csv" --pretrain-file="./target/gen-csvs/text-26.csv" --pretrain-file="./target/gen-csvs/text-27.csv" --pretrain-file="./target/gen-csvs/text-28.csv" --pretrain-file="./target/gen-csvs/text-32.csv" --pretrain-file="./target/gen-csvs/text-34.csv" --pretrain-file="./target/gen-csvs/text-35.csv" --pretrain-file="./target/gen-csvs/text-36.csv" --pretrain-file="./target/gen-csvs/text-38.csv" --pretrain-file="./target/gen-csvs/text-39.csv" --pretrain-file="./target/gen-csvs/text-40.csv" --pretrain-file="./target/gen-csvs/text-42.csv" --pretrain-file="./target/gen-csvs/text-41.csv" --pretrain-file="./target/gen-csvs/text-46.csv" --pretrain-file="./target/gen-csvs/text-44.csv" --pretrain-file="./target/gen-csvs/text-54.csv" --pretrain-file="./target/gen-csvs/text-47.csv" --pretrain-file="./target/gen-csvs/text-45.csv" --pretrain-file="./target/gen-csvs/text-43.csv" --pretrain-file="./target/gen-csvs/text-52.csv" --pretrain-file="./target/gen-csvs/text-48.csv" --pretrain-file="./target/gen-csvs/text-50.csv" --pretrain-file="./target/gen-csvs/text-51.csv" --pretrain-file="./target/gen-csvs/text-49.csv" --pretrain-file="./target/gen-csvs/text-55.csv" --pretrain-file="./target/gen-csvs/text-56.csv" --pretrain-file="./target/gen-csvs/text-57.csv" --pretrain-file="./target/gen-csvs/text-58.csv" --pretrain-file="./target/gen-csvs/text-53.csv" --pretrain-file="./target/gen-csvs/text-59.csv" --pretrain-file="./target/gen-csvs/text-60.csv" --pretrain-file="./target/gen-csvs/text-61.csv" --pretrain-file="./target/gen-csvs/text-62.csv" --pretrain-file="./target/gen-csvs/text-79.csv" --pretrain-file="./target/gen-csvs/text-78.csv" --pretrain-file="./target/gen-csvs/text-63.csv" --pretrain-file="./target/gen-csvs/text-64.csv" --pretrain-file="./target/gen-csvs/text-65.csv" --pretrain-file="./target/gen-csvs/text-67.csv" --pretrain-file="./target/gen-csvs/text-68.csv" --pretrain-file="./target/gen-csvs/text-66.csv" --pretrain-file="./target/gen-csvs/text-69.csv" --pretrain-file="./target/gen-csvs/text-70.csv" --pretrain-file="./target/gen-csvs/text-71.csv" --pretrain-file="./target/gen-csvs/text-72.csv" --pretrain-file="./target/gen-csvs/text-73.csv" --pretrain-file="./target/gen-csvs/text-74.csv" --pretrain-file="./target/gen-csvs/text-75.csv" --pretrain-file="./target/gen-csvs/text-76.csv" --pretrain-file="./target/gen-csvs/text-77.csv" --pretrain-file="./target/gen-csvs/text-89.csv" --pretrain-file="./target/gen-csvs/text-90.csv" --pretrain-file="./target/gen-csvs/text-83.csv" --pretrain-file="./target/gen-csvs/text-84.csv" --pretrain-file="./target/gen-csvs/text-85.csv" --pretrain-file="./target/gen-csvs/text-86.csv" --pretrain-file="./target/gen-csvs/text-80.csv" --pretrain-file="./target/gen-csvs/text-81.csv" --pretrain-file="./target/gen-csvs/text-82.csv" --pretrain-file="./target/gen-csvs/text-88.csv" --pretrain-file="./target/gen-csvs/text-87.csv" --pretrain-file="./target/gen-csvs/text-92.csv" --pretrain-file="./target/gen-csvs/text-98.csv" --pretrain-file="./target/gen-csvs/text-97.csv" --pretrain-file="./target/gen-csvs/text-99.csv" --pretrain-file="./target/gen-csvs/text-96.csv" --pretrain-file="./target/gen-csvs/text-91.csv" --pretrain-file="./target/gen-csvs/text-93.csv" --pretrain-file="./target/gen-csvs/text-94.csv" --pretrain-file="./target/gen-csvs/text-95.csv"
// #[tokio::main(flavor = "multi_thread", worker_threads = 4)]
fn main() -> anyhow::Result<()> {
    init_log();

    let args = args();
    info!("Startup, the args is {:?}", args);

    let rt = runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(8)
        .thread_name_fn(|| {
            static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
            let id = ATOMIC_ID.fetch_add(1, Ordering::SeqCst);
            format!("tokio-{}", id)
        })
        .disable_lifo_slot()
        .enable_all()
        .enable_metrics_poll_count_histogram()
        .metrics_poll_count_histogram_scale(HistogramScale::Log)
        .metrics_poll_count_histogram_buckets(15)
        .metrics_poll_count_histogram_resolution(Duration::from_micros(100))
        .build()
        .unwrap();

    rt.block_on(spawn_pretrain_task())
}
