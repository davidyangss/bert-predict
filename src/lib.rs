pub mod init;

#[cfg(feature = "pretrain")]
pub mod pretrain;

#[cfg(feature = "pretrain")]
pub mod prelude {
    pub use crate::init::args;
    pub use crate::init::init_log;
    pub use crate::init::CommandArgs;

    pub use crate::pretrain::chunks_timeout_of_train_files;
    pub use crate::pretrain::get_imported_lines;
    pub use crate::pretrain::pretrain_do;
    pub use crate::pretrain::pretrain_sink;
    pub use crate::pretrain::records_of_train_files;
    pub use crate::pretrain::spawn_pretrain_task;
}
