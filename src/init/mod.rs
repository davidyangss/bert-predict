use std::{
    fs,
    path::PathBuf,
    sync::{Once, OnceLock},
};

use clap::ArgMatches;
use lazy_static::lazy_static;
use log::info;
use log4rs;

static START: Once = Once::new();

pub fn init_log() {
    START.call_once(|| {
        let p: PathBuf = "config/log4rs.yaml".into();
        log4rs::init_file(p.clone(), Default::default()).unwrap();

        let absolute_path = fs::canonicalize(&p);
        info!("Init log ok, use {}", absolute_path.unwrap().display());
    });
}

#[derive(PartialEq, Eq)]
pub enum PredictType {
    PREDICT,
    TRAIN,
}

#[cfg(any(feature = "predict", not(feature = "pretrain")))]
pub const APP_TYPE: PredictType = PredictType::PREDICT;

#[cfg(feature = "pretrain")]
pub const APP_TYPE: PredictType = PredictType::TRAIN;

#[derive(Debug)]
pub struct CommandArgs {
    pub model_config: PathBuf,
    pub model_vocab: PathBuf,
    pub roberta_vocab: Option<PathBuf>,
    pub model_ot: Option<PathBuf>,
    pub pretrained_files: Option<&'static [PathBuf]>,
    pub unordered_files: usize,
    pub csv_delimiter: char,
    pub chunk_max_size: usize,
    pub chunk_timeout: u64,
    pub lower_cased: Option<bool>,
    pub strip_accents: Option<bool>,
    pub add_prefix_space: Option<bool>,
}

impl<'a> From<ArgMatches> for CommandArgs {
    fn from(matches: ArgMatches) -> Self {
        let model_config = matches.get_one::<PathBuf>("model-config").unwrap().clone();
        let model_vocab = matches.get_one::<PathBuf>("model-vocab").unwrap().clone();
        let roberta_vocab = matches
            .get_one::<PathBuf>("roberta-vocab")
            .map(|f| f.clone());
        let model_ot = matches.get_one::<PathBuf>("model-ot").map(|f| f.clone());
        let pretrained_files = matches.get_many::<PathBuf>("pretrain-file");
        let csv_delimiter = matches
            .get_one::<char>("csv-delimiter")
            .map(char::clone)
            .or(Some(','))
            .expect("csv-delimiter is required");
        let unordered_files = matches
            .get_one::<usize>("unordered-files")
            .map(usize::clone)
            .or(Some(20))
            .expect("unordered-files is required");
        let chunk_max_size = matches
            .get_one::<usize>("chunk-max-size")
            .map(usize::clone)
            .or(Some(1000))
            .expect("chunk-max-size is required");
        let chunk_timeout = matches
            .get_one::<u64>("chunk-timeout")
            .map(u64::clone)
            .or(Some(100))
            .expect("chunk-timeout is required");
        let lower_cased = matches.get_one::<bool>("lower-cased").map(bool::clone);
        let strip_accents = matches.get_one::<bool>("strip-accents").map(bool::clone);
        let add_prefix_space = matches.get_one::<bool>("add-prefix-space").map(bool::clone);

        let pretrained_files: Option<&'static [PathBuf]> = pretrained_files
            .map(|vfs| -> Vec<PathBuf> { vfs.map(|f| f.clone()).collect() })
            .map(|v| Vec::leak(v))
            .map(|f| f as &'_ [PathBuf]);

        CommandArgs {
            model_config,
            model_vocab,
            roberta_vocab,
            model_ot,
            pretrained_files,
            unordered_files,
            csv_delimiter,
            chunk_max_size,
            chunk_timeout,
            lower_cased,
            strip_accents,
            add_prefix_space,
        }
    }
}

lazy_static! {
    static ref COMMAND_ARGS: OnceLock<CommandArgs> = OnceLock::new();
}

pub fn args() -> &'static CommandArgs {
    COMMAND_ARGS.get_or_init(parse_args)
}

fn parse_args() -> CommandArgs {
    let mut cmd = clap::Command::new("predict_docs")
        .bin_name("predict_docs")
        .about("predict docs for government records service")
        .author("cn.yangss@gmail.com")
        .arg(clap::arg!(--"model-config" <PATH> "The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)")
            .value_parser(clap::value_parser!(PathBuf))
            .required(true))
        .arg(clap::arg!(--"model-vocab" <PATH> "The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)")
            .value_parser(clap::value_parser!(PathBuf))
            .required(true))
        .arg(clap::arg!(--"model-ot" <PATH> "Local subdirectory of the cache root where this resource(rust_model.ot) is saved.")
            .value_parser(clap::value_parser!(PathBuf))
            .required(true))
        .arg(clap::arg!(--"roberta-vocab" [PATH] "An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.")
            .value_parser(clap::value_parser!(PathBuf)))
        .arg(clap::arg!(--"lower-cased" [bool] "A `bool` indicating whether the tokenizer should lower case all input (in case of a lower-cased model). default: true")
            .value_parser(clap::value_parser!(bool)))
        .arg(clap::arg!(--"strip-accents" [bool] "").value_parser(clap::value_parser!(bool)))
        .arg(clap::arg!(--"add-prefix-space" [bool] "").value_parser(clap::value_parser!(bool)));

    match APP_TYPE {
        PredictType::PREDICT => cmd = cmd.bin_name("predict"),
        PredictType::TRAIN => {
            cmd = cmd.bin_name("pretrain")
                .arg(clap::arg!(--"unordered-files" [usize] "At the same time, unordered files size. default: 20")
                    .value_parser(clap::value_parser!(usize)))
                .arg(clap::arg!(--"chunk-max-size" [usize] "Buffer lines of pretrain. default: 100 * cpu core count")
                    .value_parser(clap::value_parser!(usize)))
                .arg(clap::arg!(--"chunk-timeout" [u64] "Buffer lines timeout of pretrain, unit: MS milliseconds. default: 100ms")
                    .value_parser(clap::value_parser!(u64)))
                .arg(clap::arg!(--"csv-delimiter" [char] "csv file delimiter, default = `,`")
                    .value_parser(clap::value_parser!(char)))
                .arg(clap::arg!(--"pretrain-file" <PATH> ... "array. the train files for model.")
                    .value_parser(clap::value_parser!(PathBuf))
                    .required(true));
        }
    }

    let matches = cmd.get_matches();
    CommandArgs::from(matches)
}
