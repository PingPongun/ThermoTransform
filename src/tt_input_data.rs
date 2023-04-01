use binrw::*;
use fast_float::FastFloatParser;
use ndarray::{Array, Array3, ArrayBase, Dimension, Ix3, OwnedRepr, Zip};
use ndarray::{Data, Dim};
use std::ffi::OsString;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::mem::{self, transmute};
use std::ops::{Deref, DerefMut};
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::macros::array_pows_2_3;
use crate::tt_common::*;

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

pub const SUPPORTED_FILE_EXTENSIONS : &[&str] = &["txt", "ttcf"];

pub struct ArrayBaseW<S, D>(ArrayBase<S, D>)
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
    fn eq(&self, rhs : &ArrayBaseW<S2, D>) -> bool { **self == **rhs }
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

#[binrw]
#[derive(PartialEq)]
pub struct TTInputData
{
    pub frames : u32,
    pub width :  u32,
    pub height : u32,
    #[br(map = |x: Array3W<u32>| ArrayBaseW(x.map(|&x| f32::from_bits(x) as f64)))]
    #[bw(map = |x: &Array3W<f64>| ArrayBaseW(x.map(|&x| (x as f32).to_bits())))]
    pub data :   Array3W<f64>,
}

impl TTInputData
{
    pub fn new(path : &OsString, file_state : Arc<AtomicFileState>) -> Option<Self>
    {
        // let mut exec_time = ExecutionTimeMeas::new("exec_time_input.txt");
        // exec_time.start();
        //open file
        let f : File = match File::open(path.clone())
        {
            Ok(it) => it,
            Err(_) =>
            {
                file_state.store(FileState::Error, Ordering::SeqCst);
                return None;
            }
        };
        // exec_time.stop_print("open file");
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
            rounded_frames = find_next_pows_2_3(estimated_frames);
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
            file_state.store(FileState::Error, Ordering::SeqCst);
            return None;
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
                        data :   data_transposed.into(),
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
                    file_state.store(FileState::Error, Ordering::SeqCst);
                    return None;
                }
            }
            Err(_) =>
            {
                //wrong file format
                file_state.store(FileState::Error, Ordering::SeqCst);
                return None;
            }
        }
    }
}
