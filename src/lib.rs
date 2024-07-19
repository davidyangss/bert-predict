
pub fn setup_tracing(log_level: &str, logfile: Option<&PathBuf>, display_target: Option<bool>) {
    static START: Once = Once::new();
    START.call_once(|| {
        let level = Level::from_str(log_level).expect("Invalid log level");
        // tracing_subscriber::fmt::init();

        let l = tracing_subscriber::fmt::layer::<Registry>()
            .with_thread_names(true)
            .with_target(display_target.unwrap_or(false));

        if let Some(logfile) = logfile {
            let file_appender = tracing_appender::rolling::hourly(
                logfile.parent().unwrap(),
                logfile.file_name().unwrap(),
            );
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
            let f = l
                .with_writer(non_blocking)
                .with_filter(EnvFilter::from_default_env().or(LevelFilter::from_level(level)));
            let layered = tracing_subscriber::registry().with(f);
            tracing::subscriber::set_global_default(layered)
                .expect("setting default subscriber failed");
        } else {
            let f = l.with_filter(EnvFilter::from_default_env().or(LevelFilter::from_level(level)));
            let layered = tracing_subscriber::registry().with(f);
            tracing::subscriber::set_global_default(layered)
                .expect("setting default subscriber failed");
        }

        info!(
            "OK. setup tracing, log level: {:?}, RUST_LOG={}",
            &level,
            env::var("RUST_LOG").unwrap()
        );
    });
}

lazy_static! {
    static ref TOKIO_RUNTIME: OnceLock<Runtime> = OnceLock::new();
}

pub fn runtime() -> &'static Runtime {
    init();
    info!("Commond args: {:?}", args());
    TOKIO_RUNTIME.get_or_init(|| {
        runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .max_blocking_threads(8)
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