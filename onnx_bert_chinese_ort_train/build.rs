use std::{
    env,
    path::{Path, PathBuf},
};

const ONNXRUNTIME_VERSION: &str = "1.18.0";
const ORT_ENV_ORT_DYLIB_PATH: &str = "ORT_DYLIB_PATH";

fn select_onnxruntime_lib() -> PathBuf {
    println!("cargo:rerun-if-env-changed={}", ORT_ENV_ORT_DYLIB_PATH);
    let ort_lib = env::var(ORT_ENV_ORT_DYLIB_PATH);
    if let Ok(ort_lib) = ort_lib {
        if !ort_lib.is_empty() {
            println!("cargo:warning=Using onnxruntime library: {}", ort_lib);
            return PathBuf::from(ort_lib);
        }
    }

    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let ort_lib = root
        .parent()
        .unwrap()
        .join("onnxruntime-libs")
        .join(format!("libonnxruntime.so.{}", ONNXRUNTIME_VERSION));

    if !ort_lib.exists() {
        panic!(
            "Not found onnxruntime library {}, set the ORT_DYLIB_PATH environment variable to the path to your onnxruntime.dll/libonnxruntime.so/libonnxruntime.dylib, like ORT_DYLIB_PATH=./libonnxruntime.so",
            ort_lib.display()
        );
    }
    println!(
        "cargo:warning=Using onnxruntime library: {}",
        ort_lib.display()
    );
    println!("cargo:rerun-if-changed={}", ort_lib.display());
    ort_lib
}

// RUSTFLAGS="-C link-args=-Wl,-rpath,$ORIGIN/../lib" cargo build
fn main() {
    // ldd -v target/debug/train
    // 编译时：libonnxruntime.so.{}可存在于【系统库目录、或、ORT_DYLIB_PATH指定、或、onnxruntime_lib的上级目录】
    let onnxruntime_lib = select_onnxruntime_lib();
    println!(
        "cargo:rustc-env={}={}",
        ORT_ENV_ORT_DYLIB_PATH,
        onnxruntime_lib.display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        onnxruntime_lib.parent().unwrap().display()
    );

    // readelf -d target/debug/train | grep 'R*PATH'
    // 运行时：libonnxruntime.so.{}可存在于【系统库目录、或、执行文件同级目录、或、同级lib目录、或、ORT_DYLIB_PATH指定】
    // 在Cargo.toml中配置
    // [target.x86_64-unknown-linux-gnu]
    // rustflags = [ "-Clink-args=-Wl,-rpath,\\$ORIGIN", "-Clink-args=-Wl,-rpath,\\$ORIGIN/../lib" ]
    println!("cargo:rustc-link-args=-Wl,-rpath,$ORIGIN/../lib");
    println!("cargo:rustc-link-args=-Wl,-rpath,$ORIGIN/../");
    println!(
        "cargo:rustc-link-args=-Wl,-rpath,{}",
        onnxruntime_lib.parent().unwrap().display()
    );

    println!("cargo:warning=LD_LIBRARY_PATH={}", env!("LD_LIBRARY_PATH"));
}
