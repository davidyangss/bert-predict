use std::path::{Path, PathBuf};

use ndarray::{concatenate, s, Array1, Array2, ArrayViewD, Axis};
use ort::{Allocator, CUDAExecutionProvider, Checkpoint, Session, SessionBuilder, Trainer};
use tokenizers::Tokenizer;
use tracing::info;

pub struct OrtTraining {
    trainer: Trainer,
    tokenizer: Tokenizer,
    out_trained_onnx: PathBuf,
}

impl OrtTraining {}

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

        Ok(OrtTraining {
            trainer,
            tokenizer,
            out_trained_onnx: self.out_trained_onnx,
        })
    }
}
