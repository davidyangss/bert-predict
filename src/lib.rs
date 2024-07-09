pub mod init;

pub mod prelude {
    pub use crate::init::args;
    pub use crate::init::CommandArgs;
    pub use crate::init::init_log;
}