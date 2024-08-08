use std::path::{Path, PathBuf};

use onnx_bert_chinese_ort_train_dataset::text_label::{TextLabel, TextLabelBytes as _};
use ort::{Allocator, CUDAExecutionProvider, Checkpoint, SessionBuilder, Trainer};
use tokenizers::Tokenizer;
use tracing::{info, trace};

pub struct OrtTraining {
    trainer: Trainer,
    _tokenizer: Tokenizer,
    out_trained_onnx: PathBuf,

    _training_steps: usize,
    training_batch_size: usize,
    training_sequence_length: usize,

    _ids_max_len: usize,
    attention_mask_ndarray: ndarray::Array2<IdsType>,
    token_type_ids_ndarray: ndarray::Array2<IdsType>,
}

type IdsType = i64;
type LablesType = i64;
type LossType = f32;

impl OrtTraining {
    pub fn get_trainer(&self) -> &Trainer {
        &self.trainer
    }
    pub fn step(&self, batch: &[TextLabel]) -> anyhow::Result<LossType> {
        let mut inputs =
            vec![0 as IdsType; self.training_batch_size * self.training_sequence_length];
        let mut labels = Vec::<IdsType>::with_capacity(self.training_batch_size);
        trace!(
            "step batch: shape: [{}, {}], labels = {}",
            batch.len(),
            self.training_sequence_length,
            labels.capacity()
        );
        for i in 0..batch.len() {
            let text_label = &batch[i];
            let target = inputs.as_mut_slice();
            let target = target
                [i * self.training_sequence_length..(i + 1) * self.training_sequence_length]
                .as_mut();
            text_label.bytes_into_encoding_ids::<IdsType>(text_label.id_bytes(), target);
            let label = text_label.bytes_to_encoding_ids::<LablesType>(text_label.label_bytes());
            if label.len() != 1 {
                return Err(anyhow::anyhow!("label.len() != 1, label bytes error"));
            }
            labels.extend(label);
        }

        // trace!(
        //     "step inputs: {} / {:?}",
        //     inputs.len(),
        //     inputs
        //         .iter()
        //         .filter(|i| **i != 0)
        //         .map(|i| format!("{}", hex::encode(&i.to_le_bytes())))
        //         .collect::<Vec<String>>()
        // );
        trace!("step inputs: {}, step labels: {:?}", inputs.len(), labels);

        let trainer = &self.trainer;

        let inputs: ndarray::ArrayBase<ndarray::OwnedRepr<i64>, ndarray::Dim<[usize; 2]>> = ndarray::Array2::<IdsType>::from_shape_vec(
            [self.training_batch_size, self.training_sequence_length],
            inputs,
        )
        .map_err(|e| anyhow::anyhow!("Array2::<IdsType>::from_shape_vec(inputs), error: {e}"))?;

        let labels =
            ndarray::Array1::<LablesType>::from_shape_vec([self.training_batch_size], labels)
                .map_err(|e| {
                    anyhow::anyhow!("Array1::<IdsType>::from_shape_vec(labels), error: {e}")
                })?;

        trace!("step ndarray inputs: {:?}", inputs);
        trace!("step ndarray labels: {:?}", labels);

        let inputs = ort::inputs![
            inputs.view(),
            self.attention_mask_ndarray.view(),
            self.token_type_ids_ndarray.view()
        ]
        .map_err(|e| anyhow::anyhow!("ort::inputs![inputs.view()], error: {e}"))?;
        let labels = ort::inputs![labels.view()]
            .map_err(|e| anyhow::anyhow!("ort::inputs![labels.view()], error: {e}"))?;

        let outputs = trainer
            .step(inputs, labels)
            .map_err(|e| anyhow::anyhow!("trainer.step(inputs, labels), error: {e}"))?;

        let loss = outputs[0].try_extract_scalar::<LossType>().map_err(|e| {
            anyhow::anyhow!("outputs[0].try_extract_scalar::<LossType>(), error: {e}")
        })?;
        
        if loss.is_nan() {
            return Ok(loss);
        }

        trainer
            .optimizer()
            .step()
            .map_err(|e| anyhow::anyhow!("trainer.optimizer().step(), error: {e}"))?;
        trainer
            .optimizer()
            .reset_grad()
            .map_err(|e| anyhow::anyhow!("trainer.optimizer().reset_grad(), error: {e}"))?;
        return Ok(loss);
    }

    pub fn export(&self) -> anyhow::Result<()> {
        self.trainer
            .export(&self.out_trained_onnx, ["logits"])
            .map_err(|e| anyhow::anyhow!("trainer.export, error: {e}"))?;
        Ok(())
    }

    pub fn out_trained_onnx(&self) -> &Path {
        self.out_trained_onnx.as_path()
    }
}

/// use args to create a new OrtTraining instance
#[derive(Debug, Clone)]
pub struct OrtTrainingBuilder {
    checkpoint: PathBuf,
    training_model: PathBuf,
    eval_model: PathBuf,
    optimizer_model: PathBuf,
    tokenizer_json: PathBuf,
    out_trained_onnx: PathBuf,
    optimizer_lr: f32,

    idxs_max_len: usize,

    training_steps: usize,
    training_batch_size: usize,
    training_sequence_length: usize,
}

impl Default for OrtTrainingBuilder {
    fn default() -> Self {
        Self {
            checkpoint: Default::default(),
            training_model: Default::default(),
            eval_model: Default::default(),
            optimizer_model: Default::default(),
            tokenizer_json: Default::default(),
            out_trained_onnx: Default::default(),
            optimizer_lr: Default::default(),
            training_steps: Default::default(),
            training_batch_size: Default::default(),
            training_sequence_length: Default::default(),
            idxs_max_len: Default::default(),
        }
    }
}

impl OrtTrainingBuilder {
    pub fn with_checkpoint(mut self, checkpoint: &PathBuf) -> Self {
        self.checkpoint = checkpoint.clone();
        self
    }

    pub fn with_training_model(mut self, training_model: &PathBuf) -> Self {
        self.training_model = training_model.clone();
        self
    }

    pub fn with_eval_model(mut self, eval_model: &PathBuf) -> Self {
        self.eval_model = eval_model.clone();
        self
    }

    pub fn with_optimizer_model(mut self, optimizer_model: &PathBuf) -> Self {
        self.optimizer_model = optimizer_model.clone();
        self
    }

    pub fn with_tokenizer_json(mut self, tokenizer_json: &PathBuf) -> Self {
        self.tokenizer_json = tokenizer_json.clone();
        self
    }

    pub fn with_out_trained_onnx(mut self, out_trained_onnx: &PathBuf) -> Self {
        self.out_trained_onnx = out_trained_onnx.clone();
        self
    }

    pub fn with_optimizer_lr(mut self, optimizer_lr: f32) -> Self {
        self.optimizer_lr = optimizer_lr;
        self
    }

    pub fn with_training_steps(mut self, training_steps: usize) -> Self {
        self.training_steps = training_steps;
        self
    }

    pub fn with_training_batch_size(mut self, training_batch_size: usize) -> Self {
        self.training_batch_size = training_batch_size;
        self
    }

    pub fn with_training_sequence_length(mut self, training_sequence_length: usize) -> Self {
        self.training_sequence_length = training_sequence_length;
        self
    }

    pub fn with_ids_max_len(mut self, idxs_max_len: usize) -> Self {
        self.idxs_max_len = idxs_max_len;
        self
    }

    pub fn build(self) -> anyhow::Result<OrtTraining> {
        ort::init().commit()?;

        let s = SessionBuilder::new()?;
        let p = s.with_execution_providers([CUDAExecutionProvider::default().build()])?;

        let trainer = Trainer::new(
            p,
            Allocator::default(),
            Checkpoint::load(self.checkpoint)?,
            self.training_model,
            self.eval_model,
            self.optimizer_model,
        )?;
        trainer.optimizer().set_lr(7e-5)?;

        let tokenizer = Tokenizer::from_file(self.tokenizer_json)
            .map_err(|_| anyhow::anyhow!("create tokenizer fail"))?;

        info!("Ort traning model is ready.");

        let token_type_ids =
            vec![0 as IdsType; self.training_batch_size * self.training_sequence_length];
        let token_type_ids_ndarray = ndarray::Array2::<IdsType>::from_shape_vec(
            [self.training_batch_size, self.training_sequence_length],
            token_type_ids,
        )
        .map_err(|e| {
            anyhow::anyhow!("Array2::<IdsType>::from_shape_vec(token_type_ids), error: {e}")
        })?;

        let attention_mask =
            vec![1 as IdsType; self.training_batch_size * self.training_sequence_length];
        let attention_mask_ndarray = ndarray::Array2::<IdsType>::from_shape_vec(
            [self.training_batch_size, self.training_sequence_length],
            attention_mask,
        )
        .map_err(|e| {
            anyhow::anyhow!("Array2::<IdsType>::from_shape_vec(attention_mask), error: {e}")
        })?;

        Ok(OrtTraining {
            trainer,
            _tokenizer: tokenizer,
            out_trained_onnx: self.out_trained_onnx,
            _training_steps: self.training_steps,
            training_batch_size: self.training_batch_size,
            training_sequence_length: self.training_sequence_length,
            _ids_max_len: self.idxs_max_len,
            token_type_ids_ndarray: token_type_ids_ndarray,
            attention_mask_ndarray: attention_mask_ndarray,
        })
    }
}
