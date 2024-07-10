// use std::env;

fn main() {
    // env::args().for_each(|arg| {
    //     println!("cargo:warning=arg: {:?}", arg);
    // });
    // env::vars().for_each(|arg| {
    //     println!("cargo:warning=var: {:?}", arg);
    // });
    // env::vars_os().for_each(|arg| {
    //     println!("cargo:warning=vars_os: {:?}", arg);
    // });
    // println!("cargo:warning=Running build script");

    //不可用，CARGO_BIN_NAME非编译时环境变量，所以无法在编译时添加开启features
    //在Cargo.toml中[[bin]].required-features = ["pretrain"], 在Build时，cargo build --bin pretrain --features="pretrain"
    // if let Ok(bin) = env::var("CARGO_BIN_NAME") {
    //     println!("cargo:warning=Building binary: {}", bin);

    //     if bin == "pretrain" {
    //         println!("cargo:rustc-cfg=pretrain");
    //         unimplemented!();
    //     } else if bin == "predict" {
    //         println!("cargo:rustc-cfg=predict");
    //     }
    // }
}
