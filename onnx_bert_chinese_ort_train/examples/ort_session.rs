use std::{cmp::min, usize};

use tokenizers::Tokenizer;
use tracing::info;
use yss_commons::commons_tokio::*;

use ort::Session;

type IdsType = i64;
// type LablesType = i64;
type LossType = f32;

fn main() -> anyhow::Result<()> {
    setup_tracing("INFO", None, Some(true));

    let manifest_dir = yss_commons::commons_path::cargo_manifest_dir();

    let tokinizer_json =
        manifest_dir.join("./onnx-model/google-bert-chinese/base_model/tokenizer.json");
    info!("tokinizer json is {}", tokinizer_json.display());
    let tokinizer = Tokenizer::from_file(&tokinizer_json).unwrap();

    let model_onnx = manifest_dir.join("./onnx-model/google-bert-chinese/base_model/model.onnx");
    // let model_onnx = manifest_dir.join("./onnx-model/google-bert-chinese/onnx-artifacts/training_model.onnx");
    info!("Use model file is {}", model_onnx.display());

    let session = Session::builder()?.commit_from_file(model_onnx)?;
    let session_inputs = session.inputs.as_slice();
    info!("Session inputs: {session_inputs:?}");
    let session_outputs = session.outputs.as_slice();
    info!("Session outputs: {session_outputs:?}");

    const TRAINING_BATCH_SIZE: usize = 4;
    const TRAINING_SEQUENCE_LENGTH: usize = 256;
    let token_type_ids = vec![0 as IdsType; TRAINING_BATCH_SIZE * TRAINING_SEQUENCE_LENGTH];
    let token_type_ids_ndarray = ndarray::Array2::<IdsType>::from_shape_vec(
        [TRAINING_BATCH_SIZE, TRAINING_SEQUENCE_LENGTH],
        token_type_ids,
    )
    .map_err(|e| {
        anyhow::anyhow!("Array2::<IdsType>::from_shape_vec(token_type_ids), error: {e}")
    })?;

    let attention_mask = vec![1 as IdsType; TRAINING_BATCH_SIZE * TRAINING_SEQUENCE_LENGTH];
    let attention_mask_ndarray = ndarray::Array2::<IdsType>::from_shape_vec(
        [TRAINING_BATCH_SIZE, TRAINING_SEQUENCE_LENGTH],
        attention_mask,
    )
    .map_err(|e| {
        anyhow::anyhow!("Array2::<IdsType>::from_shape_vec(attention_mask), error: {e}")
    })?;

    let mut input_ids = vec![0 as IdsType; TRAINING_BATCH_SIZE * TRAINING_SEQUENCE_LENGTH];

    let encoding = tokinizer.encode("北国风光，千里冰封，万里雪飘。望长城内外，惟余莽莽；大河上下，顿失滔滔。山舞银蛇，象，欲与天公试比高。", true).unwrap();
    // info!("type_ids: {:?}", encoding.get_type_ids());
    // info!("mask: {:?}", encoding.get_attention_mask());
    let ids = encoding
        .get_ids()
        .iter()
        .map(|i| *i as IdsType)
        .collect::<Vec<IdsType>>();
    let targets = input_ids.as_mut_slice();
    let targets = &mut targets[0..min(TRAINING_SEQUENCE_LENGTH, ids.len())];
    targets.clone_from_slice(&ids[0..min(TRAINING_SEQUENCE_LENGTH, ids.len())]);

    let input_ids_ndarray = ndarray::Array2::<IdsType>::from_shape_vec(
        [TRAINING_BATCH_SIZE, TRAINING_SEQUENCE_LENGTH],
        input_ids,
    )
    .map_err(|e| {
        anyhow::anyhow!("Array2::<IdsType>::from_shape_vec(input_ids_ndarray), error: {e}")
    })?;

    let inputs = ort::inputs![
        "input_ids" => input_ids_ndarray.view(),
        "attention_mask" => attention_mask_ndarray.view(),
        "token_type_ids" => token_type_ids_ndarray.view()
    ]
    .map_err(|e| anyhow::anyhow!("ort::inputs![inputs.view()], error: {e}"))?;

    let outputs = session
        .run(inputs)
        .map_err(|err| anyhow::anyhow!("session run error: {err:?}"))?;
    info!("Session run: {outputs:?}");

    let logits = outputs.get("logits").expect("logits is not a output?");
    info!("logits: {logits:?}");
    let loss = logits.try_extract_raw_tensor::<LossType>()
        .map_err(|e| anyhow::anyhow!("logits.try_extract_scalar::<LossType>(), error: {e}"))?;
    info!("loss={loss:?}");

    Ok(())
}
