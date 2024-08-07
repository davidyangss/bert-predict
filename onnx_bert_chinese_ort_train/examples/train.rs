use std::path::PathBuf;

use anyhow::{anyhow, Ok};
use kdam::BarExt;
use onnx_bert_chinese_ort_train::ort_training::{self};
use onnx_bert_chinese_ort_train_dataset::{csv::Record, text_label::TextLabel};

use tokenizers::Tokenizer;
use tracing::{debug, info, trace};
use yss_commons::commons_tokio::*;

fn main() -> anyhow::Result<()> {
    setup_tracing("INFO", None, Some(true));

    let manifest_dir = yss_commons::commons_path::cargo_manifest_dir();
    let json_file = manifest_dir.join("./onnx-model/google-bert-chinese/base_model/tokenizer.json");
    info!("json file is {}", json_file.display());
    let tokinizer = Tokenizer::from_file(&json_file).unwrap();

    let steps = 10;
    let ort_training = ort_training::OrtTrainingBuilder::default()
        .with_checkpoint(&PathBuf::from(
            "./onnx-model/google-bert-chinese/onnx-artifacts/checkpoint",
        ))
        .with_training_model(&PathBuf::from(
            "./onnx-model/google-bert-chinese/onnx-artifacts/training_model.onnx",
        ))
        .with_eval_model(&PathBuf::from(
            "./onnx-model/google-bert-chinese/onnx-artifacts/eval_model.onnx",
        ))
        .with_optimizer_model(&PathBuf::from(
            "./onnx-model/google-bert-chinese/onnx-artifacts/optimizer_model.onnx",
        ))
        .with_tokenizer_json(&json_file)
        .with_out_trained_onnx(&PathBuf::from(
            "./onnx-model/google-bert-chinese/onnx-artifacts/trained_model.onnx",
        ))
        .with_optimizer_lr(7e-5)
        .with_training_steps(steps)
        .with_training_batch_size(4)
        .with_training_sequence_length(256)
        .with_ids_max_len(256)
        .build()
        .map_err(|e| anyhow!("create ort training fail, error = {e}"))?;

    let mut pb = kdam::tqdm!(total = steps);

    eprintln!();
    let count = 0_usize;

    for _ in 0..steps {
        let mut v = vec![];
        for _ in 0..4 {
            let r = Record::new(Some(0_u64), "北国风光，千里冰封，万里雪飘。望长城内外，惟余莽莽；大河上下，顿失滔滔。山舞银蛇，象，欲与天公试比高。须晴日，看红装素裹，分外".to_string(), 1);
            let encoding = tokinizer.encode(r.text(), true).unwrap();
            // info!("{:?}", encoding);
            let ids = encoding.get_ids();
            let id_bytes = TextLabel::v1_style().tokenizer_ids_as_bytes(ids);
            let label_bytes = r.label().to_le_bytes();
            let label_bytes = label_bytes.map(|i| i as u32);
            let label_bytes = TextLabel::v1_style().tokenizer_ids_as_bytes(&label_bytes);
            v.push(TextLabel::v1_style().bytes(id_bytes, label_bytes));
        }

        let loss = ort_training
            .step(v.as_slice())
            .inspect_err(|e| {
                debug!("Step({count}) ort training step error, the record = {v:?}, error = {e}")
            })
            .inspect(|_| trace!("Step({count}) ort training step ok, the record = {v:?}"))?;
        pb.set_postfix(format!("Step({count})-loss={loss:.3}"));
        pb.update(1).unwrap();
    }
    eprintln!();
    let _ = kdam::term::show_cursor();
    info!("training done");

    ort_training.export()?;
    info!(
        "training done, export to {}",
        ort_training.out_trained_onnx().display()
    );

    Ok(())
}
