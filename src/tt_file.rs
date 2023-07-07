use array_base_wrapper as abw;
use binrw::io::{BufReader, NoSeek};
use binrw::*;
use fast_float::FastFloatParser;
use ndarray::{Array, Array3, ArrayBase, Dimension, Ix3, OwnedRepr, Zip};
use ndarray::{Data, Dim};
use rfd::FileDialog;
use std::ffi::OsString;
#[cfg(not(debug_assertions))]
use std::fs::remove_file;
use std::fs::File;
use std::io::{BufRead, BufWriter, Read, Seek, Write};
use std::mem::{self, transmute};
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::macros::array_pows_2_3;
use crate::tt_common::*;

#[derive(Clone, PartialEq)]
enum TTFileType
{
    TTCF,
    Text,
    Unknown,
}
#[derive(Clone)]
pub struct TTFile
{
    path :      PathBuf,
    file_type : TTFileType,
}
/// txt file with data written in human readable format, output from THERM app
/// tttf is same as txt, only extension was change for file assosication
/// ttcf compressed(zstd)& optimized(modified f32 instead f64) data written in binary format
const EXTENSIONS_TTCF : &[&str] = &["ttcf"];
const EXTENSIONS_TCTS : &[&str] = &["txt", "tcts"];
const SUPPORTED_FILE_EXTENSIONS : &[&[&str]] = &[EXTENSIONS_TTCF, EXTENSIONS_TCTS];
const HEADER_TTCF_V1 : &str = "TTCF v1";

macro_rules! ferror {
    ($file_state:ident) => {
        let _ = $file_state.compare_exchange(
            FileState::Loading,
            FileState::Error,
            Ordering::SeqCst,
            Ordering::Acquire,
        );
        return None;
    };
}

impl TTFile
{
    //////////////
    //
    //////////////
    pub fn data_store(&self, data : &TTInputData) -> Result<(), ()>
    {
        let mut new_file = self.path.clone();
        if self.file_type != TTFileType::TTCF
        {
            new_file.set_extension("ttcf");
            if let Ok(f) = File::create(new_file)
            {
                let mut fr = BufWriter::new(f);
                let mut header : [u8; 20] = [0; 20];
                header
                    .as_mut_slice()
                    .write_all(HEADER_TTCF_V1.as_bytes())
                    .expect("");
                if let Err(_) = fr.write_all(&header)
                {
                    return Err(());
                }
                let mut encoder = zstd::Encoder::new(fr, 8).unwrap();
                let _ = encoder.include_checksum(true);
                let _ = encoder.multithread(num_cpus::get().saturating_sub(2) as u32);
                let encoder = encoder.auto_finish();

                let mut encoder_buffered = NoSeek::new(BufWriter::new(encoder));
                if let Ok(_) = FileTTCF::from(data).write_le(&mut encoder_buffered)
                {
                    #[cfg(not(debug_assertions))]
                    let _ = remove_file(self.path);
                    return Ok(());
                }
                else
                {
                    return Err(());
                }
            }
            else
            {
                return Err(());
            }
        }
        else
        {
            return Ok(());
        }
    }

    //////////////
    //
    //////////////
    pub fn data_load(&mut self, file_state : Arc<AtomicFileState>) -> Option<TTInputData>
    {
        if let Ok(f) = File::open(self.path.clone())
        {
            match self.path.extension()
            {
                Some(ext)
                    if EXTENSIONS_TTCF.contains(&ext.to_string_lossy().to_string().as_str()) =>
                {
                    self.file_type = TTFileType::TTCF;
                    return Self::data_load_ttcf(f, file_state);
                }
                Some(ext)
                    if EXTENSIONS_TCTS.contains(&ext.to_string_lossy().to_string().as_str()) =>
                {
                    self.file_type = TTFileType::Text;
                    return Self::data_load_tcts(f, file_state);
                }
                Some(_) | None =>
                {
                    self.file_type = TTFileType::Unknown;
                    ferror!(file_state);
                }
            }
        }
        else
        {
            ferror!(file_state);
        }
    }

    //////////////
    //
    //////////////
    fn data_load_tcts(f : File, file_state : Arc<AtomicFileState>) -> Option<TTInputData>
    {
        // let mut exec_time = ExecutionTimeMeas::new("exec_time_input.txt");
        // exec_time.start();

        //prealocate vector that will store input data
        let mut v_f64 : Vec<f64>;
        let estimated_capacity;
        let mut rounded_frames;
        if let Ok(meta) = f.metadata()
        {
            const FILE_HEADER_LEN : usize = 20;
            const IMAGE_TOP_HEADER_LEN : usize = 1428 + 38 + 2;
            const IMAGE_ROW_HEADER_LEN : usize = 1330;
            const IMAGE_HEIGHT : usize = 288;
            const IMAGE_WIDTH : usize = 384;
            const IMAGE_PIXELS : usize = IMAGE_HEIGHT * IMAGE_WIDTH;
            const IMAGE_DATA_LEN : usize = IMAGE_PIXELS * 6;
            const IMAGE_LEN : usize = IMAGE_DATA_LEN + IMAGE_TOP_HEADER_LEN + IMAGE_ROW_HEADER_LEN;
            let estimated_frames = (meta.len() as usize - FILE_HEADER_LEN) / IMAGE_LEN;
            // rounded_frames = find_next_pows_2_3(estimated_frames);
            rounded_frames = estimated_frames;
            // exec_time.stop_print("pows_2_3 find");
            // exec_time.start();

            estimated_capacity = (rounded_frames + 1) * IMAGE_PIXELS;
            v_f64 = Vec::with_capacity(estimated_capacity);
        }
        else
        {
            rounded_frames = 0;
            estimated_capacity = 0;
            v_f64 = Vec::default();
        }
        // exec_time.stop_print("buffer alloc");
        // exec_time.start();

        let mut fr = BufReader::new(f);

        let mut not_parsing_header : bool = true;
        let mut columns : usize = 0;
        let mut rows : usize = 0;
        let mut frames : usize = 0;

        //find correct localization/ Is decimal point ',' or '.' ?
        let ff_parser;
        if let Ok(buf) = fr.fill_buf()
        {
            if buf.contains(&b',')
            {
                ff_parser = FastFloatParser::<f64>::new(b',');
            }
            else
            {
                ff_parser = FastFloatParser::<f64>::new(b'.');
            }
        }
        else
        {
            //error during reading from file
            ferror!(file_state);
        }
        // exec_time.stop_print("localisation find");
        // exec_time.start();

        //iter file and parse it float vec
        for line in fr.lines()
        {
            if file_state.load(Ordering::Relaxed) != FileState::Loading
            {
                //file selection has been changed, ongoing file reading is outdated/invalid
                return None;
            }
            match line
            {
                Ok(iline) =>
                {
                    if !iline.is_empty()
                    {
                        let mut v = iline.trim_start();
                        //unwrap will not panic, we already checked that line is not empty
                        if let Ok((_row_id, parsed_bytes)) = ff_parser.parse_partial(v)
                        {
                            //if first string in linie is number

                            if not_parsing_header
                            {
                                //parsing data
                                rows += 1;
                                columns = 0;
                                v = v.get(parsed_bytes..).unwrap();
                                loop
                                {
                                    v = v.trim_start();
                                    match ff_parser.parse_partial(v)
                                    {
                                        Ok((val, parsed_bytes)) =>
                                        {
                                            columns += 1;
                                            v_f64.push(val);
                                            v = v.get(parsed_bytes..).unwrap();
                                        }
                                        Err(_) =>
                                        {
                                            break;
                                        }
                                    };
                                }
                            }
                            else
                            {
                                //first linie with numbers are column id's; skip it
                                not_parsing_header = true;
                                columns = 0;
                                rows = 0;
                                frames += 1;
                            }
                        }
                        else
                        {
                            //line belongs to header/subheader; skip it
                            not_parsing_header = false;
                        }
                    }
                    else
                    {
                        continue;
                    }
                }
                Err(_) => continue,
            };
        }
        // exec_time.stop_print("file loaded");
        // exec_time.start();

        //checks if parsing was valid(valid file & file has not changed)
        debug_assert!((v_f64.capacity() == estimated_capacity) || (0 == estimated_capacity));
        if file_state.load(Ordering::Relaxed) != FileState::Loading
        {
            //file selection has been changed OR wrong file format, ongoing file reading is outdated/invalid
            return None;
        }
        if rounded_frames > frames
        {
            v_f64.resize(rounded_frames * rows * columns, 0.0);
            // exec_time.stop_print("buffer resize");
            // exec_time.start();
        }
        else
        {
            rounded_frames = frames;
        }
        match Array::from_shape_vec((rounded_frames, rows, columns), v_f64)
        {
            Ok(data) =>
            {
                // exec_time.stop_print("create array");
                // exec_time.start();
                if data.shape().iter().fold(true, |acc, &dim| acc & (dim > 1))
                {
                    //transpose/reverse axes to provide continous memory for across-time slices(faster ndfft)
                    let mut data_transposed_uninit =
                        Array::uninit((data.dim().2, data.dim().1, data.dim().0));
                    Zip::from(&data.reversed_axes())
                        .and(&mut data_transposed_uninit)
                        .par_for_each(|i, o| {
                            o.write(*i);
                        });
                    // exec_time.stop_print("reverse axes");
                    // exec_time.start();
                    let data_transposed = unsafe {
                        // we can now promise we have fully initialized `data_transposed`.
                        data_transposed_uninit.assume_init()
                    };
                    let ret = Some(TTInputData {
                        data :   data_transposed,
                        frames : frames as u32,
                        width :  columns as u32,
                        height : rows as u32,
                    });
                    // exec_time.stop_print("end");
                    return ret;
                }
                else
                {
                    //invalid data dimensions-> wrong file format/empty file
                    ferror!(file_state);
                }
            }
            Err(_) =>
            {
                //wrong file format
                ferror!(file_state);
            }
        }
    }

    fn data_load_ttcf(mut f : File, file_state : Arc<AtomicFileState>) -> Option<TTInputData>
    {
        let mut header : [u8; 20] = [0u8; 20];
        if let Err(_) = f.read_exact(&mut header)
        {
            ferror!(file_state);
        }
        let header_str = String::from_utf8_lossy(&header).to_string();
        let header_str = header_str.as_str().trim_end_matches("\0").trim();
        match header_str
        {
            HEADER_TTCF_V1 =>
            {
                if let Ok(decoder) = zstd::Decoder::new(f)
                {
                    if let Ok(data) = FileTTCF::read_le(&mut NoSeek::new(BufReader::new(decoder)))
                    {
                        return Some(data.into());
                    }
                    else
                    {
                        ferror!(file_state);
                    }
                }
                else
                {
                    ferror!(file_state);
                }
            }
            _ =>
            {
                ferror!(file_state);
            }
        }
    }
    //////////////
    //
    //////////////
    pub fn path(&self) -> String { self.path.to_string_lossy().to_string() }
    //////////////
    // create TTFile
    //////////////
    pub fn new_from_file_dialog() -> Option<Self>
    {
        FileDialog::new()
            .add_filter("*", &SUPPORTED_FILE_EXTENSIONS.concat())
            .add_filter("Text coded thermogram sequence", EXTENSIONS_TCTS)
            .add_filter("ThermoTransform compressed file", EXTENSIONS_TTCF)
            .pick_file()
            .map(|x| {
                Self {
                    path :      x,
                    file_type : TTFileType::Unknown,
                }
            })
    }

    pub fn new_prevalidated(path : PathBuf) -> Option<Self>
    {
        for ext in SUPPORTED_FILE_EXTENSIONS.concat()
        {
            if path.extension() == Some(OsString::from_str(ext).unwrap().as_os_str())
            {
                return Some(Self {
                    path :      path,
                    file_type : TTFileType::Unknown,
                });
            }
        }
        return None;
    }
}

impl From<OsString> for TTFile
{
    fn from(value : OsString) -> Self
    {
        Self {
            path :      value.into(),
            file_type : TTFileType::Unknown,
        }
    }
}
////////////////
////////////////
////////////////

#[binrw]
#[derive(PartialEq)]
pub struct FileTTCF
{
    pub frames : u32,
    pub width :  u32,
    pub height : u32,
    #[br(map = |x: abw::Array3W<u32>| x.map(|&x| f64_decompress(x)))]
    #[bw(map = |x: &Array3<f64>| abw::ArrayBaseW(x.map(|&x| f64_compress(x))))]
    pub data :   Array3<f64>,
}
impl Into<TTInputData> for FileTTCF
{
    fn into(self) -> TTInputData
    {
        TTInputData {
            frames : self.frames,
            width :  self.width,
            height : self.height,
            data :   self.data,
        }
    }
}
impl From<&TTInputData> for FileTTCF
{
    fn from(value : &TTInputData) -> Self
    {
        Self {
            frames : value.frames,
            width :  value.width,
            height : value.height,
            data :   value.data.clone(), //TODO !
        }
    }
}

////////////////
////////////////
////////////////
mod array_base_wrapper
{
    use super::*;
    pub struct ArrayBaseW<S, D>(pub ArrayBase<S, D>)
    where
        D : Dimension,
        S : Data,
        S::Elem : PartialEq;
    impl<A, S, D> Deref for ArrayBaseW<S, D>
    where
        S : Data<Elem = A>,
        A : PartialEq,
        D : Dimension,
    {
        type Target = ArrayBase<S, D>;

        #[inline]
        fn deref<'a>(&'a self) -> &Self::Target
        {
            let ptr = self as *const _ as *const Self::Target;
            unsafe { transmute::<_, &'a _>(&*ptr) } //TODO CHECK forget(self)
        }
    }
    impl<A, S, D> DerefMut for ArrayBaseW<S, D>
    where
        S : Data<Elem = A>,
        A : PartialEq,
        D : Dimension,
    {
        #[inline]
        fn deref_mut<'a>(&'a mut self) -> &mut Self::Target
        {
            let ptr = self as *mut _ as *mut Self::Target;
            unsafe { transmute::<_, &'a mut _>(&mut *ptr) } //TODO CHECK forget(self)
        }
    }
    impl<A, S, D> From<ArrayBase<S, D>> for ArrayBaseW<S, D>
    where
        S : Data<Elem = A>,
        A : PartialEq,
        D : Dimension,
    {
        #[inline]
        fn from(mut value : ArrayBase<S, D>) -> Self
        {
            let ptr = &mut value as *mut _ as *mut Self;
            let res = unsafe { ptr.read() };
            mem::forget(value);
            res
        }
    }
    impl<A, B, S, S2, D> PartialEq<ArrayBaseW<S2, D>> for ArrayBaseW<S, D>
    where
        A : PartialEq<B> + PartialEq<A>,
        B : PartialEq<B>,
        S : Data<Elem = A>,
        S2 : Data<Elem = B>,
        D : Dimension,
        ArrayBase<S2, D> : PartialEq<ArrayBase<S, D>>,
    {
        #[inline]
        fn eq(&self, rhs : &ArrayBaseW<S2, D>) -> bool { self.0 == rhs.0 }
    }

    pub type Array3W<T> = ArrayBaseW<OwnedRepr<T>, Ix3>;

    impl<T> BinRead for Array3W<T>
    where T : Default + for<'a> binrw::BinRead<Args<'a> = ()> + 'static + PartialEq
    {
        type Args<'a> = ();

        fn read_options<R : Read + Seek>(
            reader : &mut R,
            endian : Endian,
            _args : Self::Args<'_>,
        ) -> BinResult<Self>
        {
            let dim : (u32, u32, u32) = <_>::read_options(reader, endian, ())?;
            let dim = Dim((dim.0 as usize, dim.1 as usize, dim.2 as usize));
            let vec : Vec<T>;
            if endian == Endian::Little && cfg!(target_endian = "big")
            {
                //if endian is not native endianess
                vec = <_>::read_options(
                    reader,
                    endian,
                    VecArgs {
                        count : dim.size(),
                        inner : (),
                    },
                )?;
            }
            else
            {
                let vec_u8 : Vec<u8> = <_>::read_options(
                    reader,
                    endian,
                    VecArgs {
                        count : dim.size() * mem::size_of::<T>(),
                        inner : (),
                    },
                )?;
                vec = unsafe { transmute(vec_u8) };
            }
            let ret = unsafe { Ok(Array3::from_shape_vec_unchecked(dim, vec).into()) };

            ret
        }
    }
    impl<T> BinWrite for Array3W<T>
    where T : Default + for<'a> binrw::BinWrite<Args<'a> = ()> + 'static + PartialEq
    {
        type Args<'a> = ();

        fn write_options<W : Write + Seek>(
            &self,
            writer : &mut W,
            endian : Endian,
            _args : Self::Args<'_>,
        ) -> BinResult<()>
        {
            (
                self.dim().0 as u32,
                self.dim().1 as u32,
                self.dim().2 as u32,
            )
                .write_options(writer, endian, ())?;
            self.as_slice_memory_order()
                .unwrap()
                .write_options(writer, endian, ())?;
            Ok(())
        }
    }
}

////////////////
//Helpers
////////////////
#[inline]
fn f64_compress(x : f64) -> u32
{
    let x = ((x) as f32).to_bits();
    //expMSB-is exponent MSB, expL- is rest of exponent bits
    let (expMSB, expL) = (x & 0x40000000, x & 0x3F800000);
    //((expL << 1) | (expMSB >> 7) | (x & 0x807FFFFF)) only changes exponent bits order from expMSB,expL to expL,expMSB
    //(...)>> 10 -> zeros significand/mantysa 10 LSb
    //(reduces precision, but available precision is still about order of magnitude better than that of input data)
    //this way in most cases(probably always) first 15 bites would be 0 (better/simpler compresion by zstd)
    ((expL << 1) | (expMSB >> 7) | (x & 0x807FFFFF)) >> 10
}
#[inline]
fn f64_decompress(x : u32) -> f64
{
    //see f64_compress(), but in reversed order
    let x = x << 10;
    let (expMSB, expL) = (x & 0x00800000, x & 0x7F000000);
    let f32_bits = (expL >> 1) | (expMSB << 7) | (x & 0x807FFFFF);
    f32::from_bits(f32_bits) as f64
}

const PRIMES : &'static [usize] = array_pows_2_3!();
fn find_next_pows_2_3(val : usize) -> usize
{
    match PRIMES.binary_search(&val)
    {
        Ok(_idx) => val,
        Err(idx) =>
        {
            debug_assert!(PRIMES[idx] > val);
            debug_assert!(PRIMES[idx] < 2 * val);
            PRIMES[idx]
        }
    }
}
