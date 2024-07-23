use std::{
    env,
    path::PathBuf,
    str::FromStr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Once, OnceLock,
    },
    time::Duration,
};

use futures::Future;

use tokio::runtime::{HistogramScale, Runtime};

use tracing::{info, level_filters::LevelFilter, Level};
use tracing_subscriber::{
    filter::FilterExt,
    layer::{Filter, SubscriberExt},
    EnvFilter, Layer, Registry,
};

use lazy_static::lazy_static;

pub fn setup_tracing(log_level: &str, logfile: Option<&PathBuf>, display_target: Option<bool>) {
    fn setup_fmt_subscriber<L, F>(l: L, f: F)
    where
        L: Layer<Registry> + Send + Sync + 'static,
        F: Filter<Registry> + Send + Sync + Sized + 'static,
    {
        let r = tracing_subscriber::registry();
        let filtered = l.with_filter(f);
        let layered = r.with(filtered);
        tracing::subscriber::set_global_default(layered)
            .expect("setting default subscriber failed");
    }

    static START: Once = Once::new();
    START.call_once(|| {
        let level = Level::from_str(log_level).expect("Invalid log level");
        // tracing_subscriber::fmt::init();

        let l = tracing_subscriber::fmt::layer::<Registry>()
            .with_thread_names(true)
            .with_target(display_target.unwrap_or(false));

        let lf: LevelFilter = LevelFilter::from_level(level);
        let def = EnvFilter::from_default_env();
        let f = <EnvFilter as FilterExt<Registry>>::or(def, lf);

        if let Some(logfile) = logfile {
            let file_appender = tracing_appender::rolling::hourly(
                logfile.parent().unwrap(),
                logfile.file_name().unwrap(),
            );
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            let l = l.with_writer(non_blocking);

            setup_fmt_subscriber(l, f);
        } else {
            setup_fmt_subscriber(l, f);
        }

        info!(
            "OK. setup tracing, log level: {:?} {}",
            &level,
            env::var("RUST_LOG")
                .map(|v| { format!(", RUST_LOG={}", v) })
                .unwrap_or_default()
        );
    });
}

lazy_static! {
    static ref TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();
}

pub fn runtime() -> &'static tokio::runtime::Runtime {
    TOKIO_RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            // .worker_threads(4)
            // .max_blocking_threads(8)
            .thread_name_fn(|| {
                static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
                let id = ATOMIC_ID.fetch_add(1, Ordering::SeqCst);
                format!("tokio-{}", id)
            })
            .disable_lifo_slot()
            .enable_all()
            .enable_metrics_poll_count_histogram()
            .metrics_poll_count_histogram_scale(HistogramScale::Log)
            .metrics_poll_count_histogram_buckets(15)
            .metrics_poll_count_histogram_resolution(Duration::from_micros(100))
            .build()
            .unwrap()
    })
}

pub fn block_on<F: Future>(future: F) -> F::Output {
    runtime().block_on(future)
}
