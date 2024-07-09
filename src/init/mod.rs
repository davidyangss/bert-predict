use std::{fs, path::PathBuf, sync::{Once, OnceLock}};

use clap::ArgMatches;
use log::info;
use log4rs;
use lazy_static::lazy_static;

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
    TRAIN
}

#[cfg(any(feature = "predict", not(feature = "pretrain")))]
pub const APP_TYPE: PredictType = PredictType::PREDICT;

#[cfg(feature = "pretrain")]
pub const APP_TYPE: PredictType = PredictType::TRAIN;

#[derive(Debug)]
pub struct CommandArgs {
    pub model_config: std::path::PathBuf,
    pub model_vocab: std::path::PathBuf,
    pub roberta_vocab: Option<std::path::PathBuf>,
    pub model_ot: Option<std::path::PathBuf>,
    pub pretrained_files: Option<&'static [std::path::PathBuf]>,
    pub lower_cased: Option<bool>,
    pub strip_accents: Option<bool>,
    pub add_prefix_space: Option<bool>,
}

impl<'a> From<ArgMatches> for CommandArgs{
    fn from(matches: ArgMatches) -> Self {
        let model_config = matches.get_one::<std::path::PathBuf>("model-config").unwrap().clone();
        let model_vocab = matches.get_one::<std::path::PathBuf>("model-vocab").unwrap().clone();
        let roberta_vocab = matches.get_one::<std::path::PathBuf>("roberta-vocab").map(|f|f.clone());
        let model_ot = matches.get_one::<std::path::PathBuf>("model-ot").map(|f|f.clone());
        let pretrained_files = matches.get_many::<std::path::PathBuf>("pretrain-file");
        let lower_cased = matches.get_one::<bool>("lower-cased").map(bool::clone);
        let strip_accents = matches.get_one::<bool>("strip-accents").map(bool::clone);
        let add_prefix_space = matches.get_one::<bool>("add-prefix-space").map(bool::clone);

        let pretrained_files: Option<&'static [std::path::PathBuf]> = pretrained_files
            .map(|vfs| -> Vec<std::path::PathBuf> {
                vfs.map(|f| f.clone()).collect()
            })
            .map(|v|Vec::leak(v))
            .map(|f|f as &'_ [std::path::PathBuf]);

        CommandArgs {
            model_config,
            model_vocab,
            roberta_vocab,
            model_ot,
            pretrained_files,
            lower_cased,
            strip_accents,
            add_prefix_space,
        }

    }
}

lazy_static! {
    static ref COMMAND_ARGS: OnceLock<CommandArgs> = OnceLock::new();
}

// cargo run --bin pretrain --features="pretrain" -- --model-config="/home/yangss/Downloads/config.json" --model-vocab="/home/yangss/Downloads/vocab.txt" --model-ot="/home/yangss/Downloads/rust_model.ot" --pretrain-file="/home/yangss/Downloads/comments.csv" --pretrain-file="/home/yangss/Downloads/comments1.csv"
fn parse_args() -> CommandArgs {
    let mut cmd = clap::Command::new("predict_docs")
        .bin_name("predict_docs")
        .about("predict docs for government records service")
        .author("cn.yangss@gmail.com")
        .arg(clap::arg!(--"model-config" <PATH> "The `ResourceProvider` pointing to the model configuration to load (e.g. config.json)")
            .value_parser(clap::value_parser!(std::path::PathBuf))
            .required(true))
        .arg(clap::arg!(--"model-vocab" <PATH> "The `ResourceProvider` pointing to the tokenizer's vocabulary to load (e.g.  vocab.txt/vocab.json)")
            .value_parser(clap::value_parser!(std::path::PathBuf))
            .required(true))
        .arg(clap::arg!(--"model-ot" <PATH> "Local subdirectory of the cache root where this resource(rust_model.ot) is saved.")
                .value_parser(clap::value_parser!(std::path::PathBuf))
                .required(true))
        .arg(clap::arg!(--"roberta-vocab" [PATH] "An optional `ResourceProvider` pointing to the tokenizer's merge file to load (e.g.  merges.txt), needed only for Roberta.")
            .value_parser(clap::value_parser!(std::path::PathBuf)))
        .arg(clap::arg!(--"lower-cased" [bool] "A `bool` indicating whether the tokenizer should lower case all input (in case of a lower-cased model). default: true")
            .value_parser(clap::value_parser!(bool)))
        .arg(clap::arg!(--"strip-accents" [bool] "").value_parser(clap::value_parser!(bool)))
        .arg(clap::arg!(--"add-prefix-space" [bool] "").value_parser(clap::value_parser!(bool)));

    match APP_TYPE {
        PredictType::PREDICT => {
            cmd = cmd.bin_name("predict")
        },
        PredictType::TRAIN => {
            cmd = cmd.bin_name("pretrain")
                .arg(clap::arg!(--"pretrain-file" <PATH> ... "array. the train files for model.")
                    .value_parser(clap::value_parser!(std::path::PathBuf))
                    .required(true));
        }
    }

    let matches = cmd.get_matches();
    CommandArgs::from(matches)
}

pub fn args() -> &'static CommandArgs {
    COMMAND_ARGS.get_or_init(parse_args)
}
