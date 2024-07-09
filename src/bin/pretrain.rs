use log::info;
use bert::prelude::*;


fn main() {
    init_log();

    let args = args();
    info!("Startup, the args is {:?}", args);
}


