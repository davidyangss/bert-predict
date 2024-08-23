use std::{cmp::min, path::PathBuf, ptr::NonNull};

use onnx_bert_chinese_ort_train::ort_training::{self, OrtTraining};

use ort::{char_p_to_string, sys::OrtTrainingSession, trainsys, Allocator, Error};
use tokenizers::Tokenizer;
use tracing::info;
use yss_commons::commons_tokio::*;

const STEPS: usize = 10;
const TRAINING_BATCH_SIZE: usize = 4;
const TRAINING_SEQUENCE_LENGTH: usize = 256;

type IdsType = i64;
// type LablesType = i64;
// type LossType = f32;

fn main() -> anyhow::Result<()> {
    setup_tracing("INFO", None, Some(true));

    let manifest_dir = yss_commons::commons_path::cargo_manifest_dir();
    let json_file = manifest_dir.join("./onnx-model/google-bert-chinese/base_model/tokenizer.json");
    info!("json file is {}", json_file.display());
    let tokinizer = Tokenizer::from_file(&json_file).unwrap();

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
        .with_training_steps(STEPS)
        .with_training_batch_size(4)
        .with_training_sequence_length(256)
        .with_ids_max_len(256)
        .build()
        .map_err(|e| anyhow::anyhow!("create ort training fail, error = {e}"))?;

    let ort_training_session_ptr = ort_training.get_trainer().ort_training_session_ptr();
    let train_output_names = ort_training.get_trainer().train_output_names();
    info!("train_output_names: {train_output_names:?}");

    print_output_names(ort_training_session_ptr)?;

    do_training(tokinizer, ort_training)?;

    Ok(())
}

fn do_training(tokinizer: Tokenizer, ort_training: OrtTraining) -> anyhow::Result<()> {
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
            input_ids_ndarray.view(),
            attention_mask_ndarray.view(),
            token_type_ids_ndarray.view()
        ]
        .map_err(|e| anyhow::anyhow!("ort::inputs![inputs.view()], error: {e}"))?;
    let labels_ndarray = ndarray::Array1::<IdsType>::from_vec(vec![1; TRAINING_BATCH_SIZE]);
    let labels = ort::inputs![labels_ndarray]
        .map_err(|e| anyhow::anyhow!("ort::inputs![labels.view()], error: {e}"))?;

    ort_training.get_trainer().step(inputs, labels)?;
    info!("training done");

    Ok(())
}

fn print_output_names(ort_training_session_ptr: NonNull<OrtTrainingSession>) -> anyhow::Result<()> {
    let allocator = Allocator::default();

    let mut train_output_len = 0;
    trainsys![unsafe TrainingSessionGetTrainingModelOutputCount(ort_training_session_ptr.as_ptr(), &mut train_output_len) -> Error::CreateSession];
    let train_output_names = (0..train_output_len)
			.map(|i| {
				let mut name_bytes: *mut std::ffi::c_char = std::ptr::null_mut();
				trainsys![unsafe TrainingSessionGetTrainingModelOutputName(ort_training_session_ptr.as_ptr(), i, allocator.ptr.as_ptr(), &mut name_bytes) -> Error::CreateSession];
				let name = match char_p_to_string(name_bytes) {
					Ok(name) => name,
					Err(e) => {
						unsafe { allocator.free(name_bytes) };
						return Err(e);
					}
				};
				unsafe { allocator.free(name_bytes) };
				Ok(name)
			})
			.collect::<Result<Vec<String>, ort::Error>>();
    info!("print_output_names: {train_output_names:?}");
    Ok(())
}
