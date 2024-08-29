use std::{
    io::IoSlice,
    mem::{self},
    pin::Pin,
    sync::atomic::AtomicUsize,
    task::{Context, Poll},
};

use anyhow::{anyhow, Ok};
use futures::Stream;
use tokio::io::{
    self, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter, ReadBuf,
};
use tracing::trace;

pin_project_lite::pin_project! {
    pub struct DatasetReader<T, R>
    where R: AsyncRead{
        #[pin]
        buf_reader: BufReader<R>,
        reading_item_style: T,
        _phantom: std::marker::PhantomData<T>
    }
}

impl<R: AsyncRead + Unpin> DatasetReader<TextLabel, R> {
    pub async fn new(read: R) -> anyhow::Result<Self> {
        const VERSION_TYPE_SIZE: usize = mem::size_of::<TLLen>();
        let mut buf_reader = BufReader::new(read);
        let mut version = [0u8; VERSION_TYPE_SIZE];
        let exact_r = buf_reader.read_exact(&mut version).await;
        match exact_r {
            Err(e) => return Err(anyhow!("excepted read version, but err: {e}")),
            Result::Ok(size) if size != VERSION_TYPE_SIZE => {
                return Err(anyhow!(
                    "excepted version type length: {VERSION_TYPE_SIZE}, but read size = {size}"
                ))
            }
            Result::Ok(_) => {}
        }

        let version = TLLen::from_be_bytes(version);
        let style = match version {
            V1_MAGIC => TextLabel::V1(V1TextLabel::default()),
            _ => anyhow::bail!("unknown version: {:#X}", version),
        };

        let s = Self {
            buf_reader,
            reading_item_style: style,
            _phantom: std::marker::PhantomData,
        };

        Ok(s)
    }

    pub fn into_stream(self: Self) -> impl Stream<Item = anyhow::Result<TextLabel>> + Unpin {
        let s = futures::stream::unfold(Box::pin(self), |mut this| async move {
            let mut tl = this.reading_item_style.to_be_filled();
            // let r = this.buf_reader.get_mut(); // 坑，直接是EOF给你看
            // let r = this.as_mut().get_mut(); // Ok
            let r = this.as_mut().project().buf_reader.get_mut(); // Ok
            let fr = tl.fill_by_read(r).await;
            match fr {
                Some(Err(e)) => Some((Err(e), this)),
                Some(Result::Ok(_)) => Some((Ok(tl), this)),
                None => None,
            }
        });
        Box::pin(s)
    }

    pub fn item_style(&self) -> &TextLabel {
        &self.reading_item_style
    }
}

impl<R: AsyncRead> AsyncRead for DatasetReader<TextLabel, R> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        self.project().buf_reader.poll_read(cx, buf)
    }
}

pin_project_lite::pin_project! {
    pub struct DatasetWriter<T, W: AsyncWrite> {
        #[pin]
        buf_writer: BufWriter<W>,
        writing_item_style: T,
        size_of_written: AtomicUsize,
        ids_max_len: AtomicUsize,
        _phantom: std::marker::PhantomData<T>,
    }
}

impl<W: AsyncWrite + Unpin> DatasetWriter<TextLabel, W> {
    pub async fn new(write: W, writing_item_style: TextLabel) -> anyhow::Result<Self> {
        let mut buf_writer = BufWriter::new(write);
        buf_writer
            .write_all(writing_item_style.version_bytes().as_slice())
            .await?;

        let size_of_written = AtomicUsize::new(writing_item_style.version_bytes().len());
        let s = Self {
            buf_writer: buf_writer,
            writing_item_style: writing_item_style,
            size_of_written: size_of_written,
            ids_max_len: AtomicUsize::new(0),
            _phantom: std::marker::PhantomData,
        };

        Ok(s)
    }

    pub fn style_bytes(&self, id_bytes: Vec<u8>, label_bytes: Vec<u8>) -> TextLabel {
        self.writing_item_style.bytes(id_bytes, label_bytes)
    }

    pub async fn write_all<'a>(&'a mut self, buf: &'a [TextLabel]) -> anyhow::Result<usize> {
        let b = buf
            .iter()
            .fold(Vec::<IoSlice>::new(), |acc, tl| tl.to_ioslices(acc));
        // 不重写poll_write_vectored，躺给你看。只写第一个。而poll_write_vectored🈶默认实现
        let size = self.write_vectored(b.as_slice()).await?;
        Ok(size)
    }

    pub fn size_of_written(&self) -> usize {
        self.size_of_written
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn item_style(&self) -> &TextLabel {
        &self.writing_item_style
    }

    pub fn update(&mut self, ids_max_len: Option<usize>) {
        if None == ids_max_len {
            return;
        }
        let ids_max_len = ids_max_len.unwrap();
        loop {
            let old = self.ids_max_len.load(std::sync::atomic::Ordering::Relaxed);
            if old >= ids_max_len {
                break;
            }
            if self
                .ids_max_len
                .compare_exchange(
                    old,
                    ids_max_len,
                    std::sync::atomic::Ordering::Relaxed,
                    std::sync::atomic::Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
    }
    pub fn ids_max_len(&self) -> usize {
        self.ids_max_len.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn count_size_of_poll(
    po: Poll<Result<usize, std::io::Error>>,
    count: &mut AtomicUsize,
) -> Poll<Result<usize, std::io::Error>> {
    match po {
        std::task::Poll::Ready(Result::Ok(size)) => {
            trace!("write size: {}", size);
            count.fetch_add(size, std::sync::atomic::Ordering::Relaxed);
            std::task::Poll::Ready(Result::Ok(size))
        }
        std::task::Poll::Ready(r) => std::task::Poll::Ready(r),
        Poll::Pending => Poll::Pending,
    }
}

impl<W: AsyncWrite> AsyncWrite for DatasetWriter<TextLabel, W> {
    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        trace!(
            "AsyncWrite buf: {} / {:?}",
            bufs.len(),
            bufs.iter().fold(Vec::<String>::new(), |mut acc, io_slice| {
                let i = hex::encode(io_slice.as_ref());
                acc.push(i);
                acc
            })
        );
        let pj = self.project();
        let po = pj.buf_writer.poll_write_vectored(cx, bufs);
        count_size_of_poll(po, pj.size_of_written)
    }

    fn poll_write(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        trace!("AsyncWrite buf: {} / {}", buf.len(), hex::encode(buf));
        let pj = self.project();
        let po = pj.buf_writer.poll_write(cx, buf);
        count_size_of_poll(po, pj.size_of_written)
    }

    fn poll_flush(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context,
    ) -> std::task::Poll<std::io::Result<()>> {
        trace!("AsyncWrite poll flush");
        self.project().buf_writer.poll_flush(cx)
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context,
    ) -> std::task::Poll<std::io::Result<()>> {
        trace!("AsyncWrite poll poll_shutdown");
        self.project().buf_writer.poll_shutdown(cx)
    }
}

pub trait TextLabelBytes {
    fn version(&self) -> TLLen;
    fn version_bytes(&self) -> Vec<u8> {
        let v = self.version();
        v.to_be_bytes().to_vec()
    }
    fn id_bytes(&self) -> &[u8];
    fn label_bytes(&self) -> &[u8];
    fn to_ioslices<'a>(&'a self, v: Vec<IoSlice<'a>>) -> Vec<IoSlice<'a>>;

    /// None == EOF
    // async fn fill_by_read<R: AsyncRead + Unpin>(&mut self, r: &mut R) -> Option<anyhow::Result<()>>;
    fn fill_by_read<'a, R: AsyncRead + Unpin>(
        &'a mut self,
        r: &'a mut R,
    ) -> impl std::future::Future<Output = Option<anyhow::Result<()>>> + Unpin + 'a;
}

const V1_MAGIC: u64 = 0x00000000C0FFEEu64;

#[derive(Debug)]
pub enum TextLabel {
    V1(V1TextLabel),
}

fn tokenizer_id_into<T: TryFrom<u32>>(id: &u32) -> anyhow::Result<T> {
    let r = TryInto::<T>::try_into(*id)
        .map_err(|_| anyhow!("tokenizer id into u{} fail", std::mem::size_of::<T>()))?;
    Ok(r)
}

impl TextLabel {
    pub fn v1_style() -> Self {
        Self::V1(V1TextLabel::default())
    }
    pub fn to_be_filled(&self) -> Self {
        match self {
            Self::V1(_) => Self::V1(V1TextLabel::default()),
        }
    }
    pub fn bytes(&self, id_bytes: Vec<u8>, label_bytes: Vec<u8>) -> Self {
        match self {
            Self::V1(_) => Self::V1(V1TextLabel::with_bytes(id_bytes, label_bytes)),
        }
    }

    pub fn tokenizer_ids_as_bytes(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .flat_map(|id| match self {
                Self::V1(_) => tokenizer_id_into::<u16>(id).unwrap().to_le_bytes(),
            })
            .collect()
    }

    pub fn bytes_to_encoding_ids<T: From<u16> + Default>(&self, bytes: &[u8]) -> Vec<T> {
        let mut targets = Vec::<T>::with_capacity(bytes.len() / 2);
        targets.resize_with(targets.capacity(), || <T as Default>::default());
        self.bytes_into_encoding_ids(bytes, targets.as_mut_slice());
        targets
    }

    pub fn bytes_into_encoding_ids<T: From<u16>>(&self, bytes: &[u8], targets: &mut [T]) {
        match self {
            Self::V1(_) => bytes
                .chunks_exact(2)
                .map(|c| {
                    TryInto::<T>::try_into(u16::from_le_bytes(
                        c.try_into().expect("can not get a u16 bytes"),
                    ))
                    .expect(&format!("u16 can not into u{}", std::mem::size_of::<T>()))
                })
                .enumerate()
                .for_each(move |(i, t)| {
                    if i >= targets.len() {
                        trace!("When bytes into encoding ids, targets too short, drop some data[{i}..]");
                        return;
                    }
                    targets[i] = t;
                }),
        }
    }

    /// v1: ids为u32的数组，每个u32转为u16的bytes
    pub fn tokenizer_ids_len(&self, bytes: &[u8]) -> usize {
        match self {
            Self::V1(_) => bytes.len() / 2,
        }
    }
}

#[test]
fn test_ids_bytes() {
    let ids = [567_u32, 123, 456, 789];
    println!(
        "ids: {}",
        ids.iter()
            .map(|b| hex::encode(b.to_le_bytes()))
            .collect::<Vec<String>>()
            .join(",")
    );
    let tl = TextLabel::v1_style();
    let ids_bytes = tl.tokenizer_ids_as_bytes(&ids);
    println!(
        "ids_bytes: {}",
        ids_bytes
            .chunks_exact(2)
            .map(|b| hex::encode(b))
            .collect::<Vec<String>>()
            .join(",")
    );
    let ids_2 = tl.bytes_to_encoding_ids::<u32>(&ids_bytes);
    println!(
        "ids_2: {}",
        ids_2
            .iter()
            .map(|b| hex::encode(b.to_le_bytes()))
            .collect::<Vec<String>>()
            .join(",")
    );
    assert_eq!(ids.to_vec(), ids_2);

    let mut ids_3 = vec![0u32; ids_2.len()];
    tl.bytes_into_encoding_ids(&ids_bytes, ids_3.as_mut_slice());
    assert_eq!(ids_3, ids_2);
}

impl TextLabelBytes for TextLabel {
    fn id_bytes(&self) -> &[u8] {
        match self {
            Self::V1(v) => v.id_bytes(),
        }
    }
    fn label_bytes(&self) -> &[u8] {
        match self {
            Self::V1(v) => v.label_bytes(),
        }
    }
    fn version(&self) -> TLLen {
        match self {
            Self::V1(v) => v.version(),
        }
    }

    fn to_ioslices<'a>(&'a self, v: Vec<IoSlice<'a>>) -> Vec<IoSlice<'a>> {
        match self {
            Self::V1(v1) => v1.to_ioslices(v),
        }
    }

    fn fill_by_read<'a, R: AsyncRead + Unpin>(
        &'a mut self,
        r: &'a mut R,
    ) -> impl std::future::Future<Output = Option<anyhow::Result<()>>> + 'a {
        match self {
            Self::V1(v) => v.fill_by_read(r),
        }
    }
}

type TLLen = u64; //长度的类型
#[derive(Debug)]
pub struct V1TextLabel {
    id_label_length_bytes: Vec<u8>,
    id_length_bytes: Vec<u8>,
    id_bytes: Vec<u8>,
    label_bytes: Vec<u8>,
}

impl Default for V1TextLabel {
    fn default() -> Self {
        Self::with_bytes(Vec::with_capacity(0), Vec::with_capacity(0))
    }
}

impl V1TextLabel {
    fn with_bytes(id_bytes: Vec<u8>, label_bytes: Vec<u8>) -> Self {
        Self {
            id_label_length_bytes: TryInto::<TLLen>::try_into(id_bytes.len() + label_bytes.len())
                .unwrap()
                .to_be_bytes()
                .to_vec(),
            id_length_bytes: TryInto::<TLLen>::try_into(id_bytes.len())
                .unwrap()
                .to_be_bytes()
                .to_vec(),
            id_bytes: id_bytes,
            label_bytes: label_bytes,
        }
    }
}

impl TextLabelBytes for V1TextLabel {
    fn id_bytes(&self) -> &[u8] {
        &self.id_bytes
    }

    fn label_bytes(&self) -> &[u8] {
        &self.label_bytes
    }

    fn version(&self) -> TLLen {
        V1_MAGIC
    }

    fn to_ioslices<'a>(&'a self, mut v: Vec<IoSlice<'a>>) -> Vec<IoSlice<'a>> {
        v.push(IoSlice::new(self.id_label_length_bytes.as_slice())); // id + label bytes 长度
        v.push(IoSlice::new(self.id_length_bytes.as_slice())); // id bytes 长度
        v.push(IoSlice::new(self.id_bytes.as_slice())); // id bytes
        v.push(IoSlice::new(self.label_bytes.as_slice())); // label bytes
        v
    }
    fn fill_by_read<'a, R: AsyncRead + Unpin>(
        &'a mut self,
        r: &'a mut R,
    ) -> impl std::future::Future<Output = Option<anyhow::Result<()>>> + Unpin + 'a {
        let fut = async move {
            const TLLEN_SIZE: usize = mem::size_of::<TLLen>();
            const LENGTH_SIZE: usize = 2 * TLLEN_SIZE;
            // 总长度
            let mut lengths = [0u8; LENGTH_SIZE]; //读出两个长度
            let exact_r = r.read_exact(&mut lengths).await;
            match exact_r {
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return None,
                Err(e) => return Some(Err(e.into())),
                Result::Ok(size) if size != (LENGTH_SIZE) => {
                    return Some(Err(anyhow!(
                        "excepted lengths-size: {LENGTH_SIZE}, but read size = {size}"
                    )))
                }
                Result::Ok(_) => {}
            }

            let id_label_len: [u8; TLLEN_SIZE] = (&lengths[0..TLLEN_SIZE])
                .try_into()
                .expect(&format!("lengths[0..{TLLEN_SIZE}] to [u8; {TLLEN_SIZE}]"));
            let id_label_len = TLLen::from_be_bytes(id_label_len) as usize;
            if id_label_len == 0 {
                return Some(Err(anyhow!("expected id_label_len > 0, but == 0")));
            }

            let id_len: [u8; TLLEN_SIZE] = (&lengths[TLLEN_SIZE..])
                .try_into()
                .expect(&format!("lengths[{TLLEN_SIZE}..] to [u8; {TLLEN_SIZE}]"));
            let id_len = TLLen::from_be_bytes(id_len) as usize;
            if id_len == 0 {
                return Some(Err(anyhow!("expected id_len > 0, but == 0")));
            }

            let mut id_label_bytes = Vec::with_capacity(id_label_len);
            id_label_bytes.resize(id_label_len, 0);
            let exact_r = r.read_exact(&mut id_label_bytes).await;
            match exact_r {
                Err(e) => {
                    return Some(Err(anyhow!(
                        "expected read bytes:({id_label_len}), but error: {e}"
                    )))
                }
                Result::Ok(size) if size != id_label_len => {
                    return Some(Err(anyhow!(
                        "excepted read bytes length: {id_label_len} / capacity: {}, but read size = {size}", id_label_bytes.capacity()
                    )))
                }
                Result::Ok(_) => {}
            }

            let id_bytes = id_label_bytes[0..id_len].to_vec();
            let label_bytes = id_label_bytes[id_len..].to_vec();
            *self = Self::with_bytes(id_bytes, label_bytes);

            Some(Ok(()))
        };
        Box::pin(fut)
    }
}

#[tokio::test]
async fn test_text_label() {
    use std::io::Cursor;

    let v = Vec::new();
    let tl = TextLabel::v1_style().bytes(b"123".to_vec(), b"456".to_vec());
    let v = tl.to_ioslices(v);
    let bytes = v.iter().fold(Vec::new(), |mut acc, io_slice| {
        acc.extend_from_slice(io_slice);
        acc
    });
    println!("bytes: {}", bytes.len());
    let mut bytes_slice = Cursor::new(bytes);

    let mut tl2 = TextLabel::V1(V1TextLabel::default());
    let fr = TextLabel::fill_by_read(&mut tl2, &mut bytes_slice).await;
    if let Some(Err(e)) = &fr {
        println!("error: {:?}", e);
    }
    // println!("remain length = {}", bytes_slice.remaining_slice().len());
    assert!(fr.is_some() && fr.unwrap().is_ok());
    assert_eq!(tl.id_bytes(), tl2.id_bytes());
    assert_eq!(tl.label_bytes(), tl2.label_bytes());
}

#[test]
fn test_tokinizer() {
    use crate::csv::Record;
    use tokenizers::Tokenizer;

    let r = Record::new(Some(0_u64), "北国风光，千里冰封，万里雪飘。望长城内外，惟余莽莽；大河上下，顿失滔滔。山舞银蛇，象，欲与天公试比高。须晴日，看红装素裹，分外".to_string(), 1);

    let manifest_dir = yss_commons::commons_path::cargo_manifest_dir();
    let json_file = manifest_dir.join("onnx-model/google-bert-chinese/base_model/tokenizer.json");
    println!("json file is {}", json_file.display());
    let tokinizer = Tokenizer::from_file(&json_file).unwrap();
    // let encoding = tokinizer.encode(r.text(), false).unwrap();
    // let encoding = tokinizer.encode(r.text(), true).unwrap(); // true, add [CLS] [SEP]等
    let encoding = tokinizer.encode((r.text(), "子句2"), true).unwrap();
    println!("{:?}", encoding);
    let id_bytes: Vec<u8> = encoding
        .get_ids()
        .iter()
        .flat_map(|c| (*c as u16).to_be_bytes())
        .collect();
    let id_bytes_hex_string = hex::encode(&id_bytes);
    let label_bytes: Vec<u8> = tokinizer
        .encode(r.label().to_string(), false)
        .map_err(|e| anyhow::anyhow!("{:?}", e))
        .unwrap()
        .get_ids()
        .iter()
        .flat_map(|c| (*c as u16).to_be_bytes())
        .collect();
    let label_bytes_hex_string = hex::encode(&label_bytes);
    println!("{} {:?}", id_bytes.len(), id_bytes_hex_string);
    println!("{} {:?}", label_bytes.len(), label_bytes_hex_string);
}

// cargo test --package onnx-ort-train-sentiment-dataset --lib -- text_label::test_writer_reader --exact --show-output
#[tokio::test]
async fn test_writer_reader() {
    use crate::csv::Record;
    use futures::{stream, StreamExt};
    use std::io::Cursor;

    use std::{
        future::ready,
        sync::{Arc, RwLock},
    };

    async fn fut() -> anyhow::Result<()> {
        let rec1 = Record::new(Some(0_u64), "北国风光，千里冰封，万里雪飘。望长城内外，惟余莽莽；大河上下，顿失滔滔。山舞银蛇，象，欲与天公试比高。须晴日，看红装素裹，分外".to_string(), 1);
        let rec2 = Record::new(
            Some(0_u64),
            "江山如此多娇，引无数英雄竞折腰。惜秦皇汉武，略输文采；唐宗宋祖，稍逊风骚。一代天骄"
                .to_string(),
            0,
        );
        let rec3 = Record::new(
            Some(0_u64),
            "成吉思汗，只识弯弓射大雕。俱往矣，数风流人物，还看今朝".to_string(),
            1,
        );

        let test_records = vec![rec1, rec2, rec3];

        let records = test_records.clone();
        let bytes = Vec::<u8>::new();
        let writer = DatasetWriter::new(bytes, TextLabel::v1_style()).await?;
        let writer = &Arc::new(RwLock::new(writer));
        let s = stream::iter(records)
            .map(move |r| -> anyhow::Result<TextLabel> {
                let rwri = writer.clone();
                let rwri = rwri.read().unwrap();
                anyhow::Result::<_, anyhow::Error>::Ok(rwri.style_bytes(
                    r.text().as_bytes().to_vec(),
                    r.label().to_string().as_bytes().to_vec(),
                ))
            })
            .filter_map(|r| async {
                let o = r.ok();
                println!("text label is {:?}", &o);
                o
            })
            .chunks(10)
            .then(|vec| async move {
                let wwri = writer.clone();
                let mut wwri = wwri.write().unwrap();
                println!("Vec<text-label> size {:?}", vec.len());
                wwri.write_all(&vec).await?;
                // wwri.flush().await?; // 不要flush，否则buffer会被清空

                Ok(vec.len())
            });
        s.for_each(|_| ready(())).await;

        let wwri = writer.clone();
        let rwri = wwri.write().unwrap();
        let wbuf = rwri.buf_writer.buffer();
        println!("writer done, bytes = {}", wbuf.len());

        let c = Cursor::new(wbuf);
        // println!(
        //     "read from cursor, remaining_slice: {}, {:?}",
        //     c.remaining_slice().len(),
        //     c.remaining_slice()
        // );
        let r = DatasetReader::new(c).await?;
        println!("version = {:?}", r.reading_item_style.version_bytes());
        assert_eq!(
            r.reading_item_style.version(),
            TextLabel::v1_style().version()
        );

        let v = r
            .into_stream()
            .filter_map(|r| async move {
                println!("filter_map = {:?}", r);
                assert!(r.is_ok());
                r.ok()
            })
            .map(|i| {
                let id = i.id_bytes();
                println!("id_bytes: {:?}", id);
                let id = std::str::from_utf8(id).unwrap().to_owned();
                let label = i.label_bytes();
                let label = std::str::from_utf8(label).unwrap().to_owned();
                Record::new(Some(0), id, label.parse().unwrap())
            })
            .inspect(|t| {
                println!("Item: {}", t);
            })
            .collect::<Vec<_>>()
            .await;
        assert_eq!(v, test_records);

        Ok(())
    }

    let r = fut().await;
    if let Err(e) = r {
        println!("error: {:?}", e);
        panic!("error: {:?}", e);
    }
}
