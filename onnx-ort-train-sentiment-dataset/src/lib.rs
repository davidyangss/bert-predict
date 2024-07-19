pub mod csv;

use std::{
    path::PathBuf,
    sync::{Once, OnceLock},
};

use clap::Parser;
use lazy_static::lazy_static;

pub mod prelude {
    pub use crate::args;
    pub use crate::csv::spawn_dataset_task;
    pub use crate::init;

    pub use tracing::debug;
    pub use tracing::error;
    pub use tracing::info;
    pub use tracing::trace;
    pub use tracing::warn;
}

use prelude::*;
use yss_commons::commons_tokio::setup_tracing;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about)]
pub struct Args {
    /// csv files
    #[clap(short, long, value_parser, num_args = 1.., value_delimiter = ' ')]
    pub csvs: Vec<PathBuf>,

    /// csv delimiter, default = ,
    #[arg(long, default_value = ",")]
    pub csv_delimiter: char,

    #[arg(short = 'l', long, default_value = "INFO")]
    pub log_level: String,

    /// At the same time, unordered files size. default: 20
    #[arg(long, default_value = "20")]
    pub unordered: usize,

    /// It is max size of chunk, when import csv line
    #[arg(long, default_value = "1000")]
    chunk_max_size: usize,

    /// It is timeout of chunk, when import csv line. unit: milliseconds
    #[arg(long, default_value = "500")]
    chunk_timeout: u64,
}

lazy_static! {
    static ref COMMAND_ARGS: OnceLock<Args> = OnceLock::new();
}

pub fn args() -> &'static Args {
    COMMAND_ARGS.get_or_init(Args::parse)
}

pub fn init() {
    static START: Once = Once::new();
    START.call_once(|| {
        setup_tracing(&args().log_level, None, None);
        info!("OK. command args: {:?}", args());
    });
}
