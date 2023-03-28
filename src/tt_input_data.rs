use fast_float::FastFloatParser;
use ndarray::{Array, Array3};
use std::ffi::OsString;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::tt_common::*;

pub const SUPPORTED_FILE_EXTENSIONS : &[&str] = &["txt"];
#[derive(PartialEq)]
pub struct TTInputData
{
    pub data : Array3<f64>,
}

impl TTInputData
{
    pub fn new(path : &OsString, file_state : Arc<AtomicFileState>) -> Option<Self>
    {
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

        //prealocate vector that will store input data
        let mut v_f64 : Vec<f64>;
        let estimated_capacity;
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
            estimated_capacity = (estimated_frames + 1) * IMAGE_PIXELS;
            v_f64 = Vec::with_capacity(estimated_capacity);
        }
        else
        {
            estimated_capacity = 0;
            v_f64 = Vec::default();
        }

        let mut fr = BufReader::new(f);

        let mut not_parsing_header : bool = true;
        let mut columns : usize = 0;
        let mut rows : usize = 0;
        let mut depths : usize = 0;

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
                                depths += 1;
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

        //checks if parsing was valid(valid file & file has not changed)
        debug_assert!((v_f64.capacity() == estimated_capacity) || (0 == estimated_capacity));
        if file_state.load(Ordering::Relaxed) != FileState::Loading
        {
            //file selection has been changed OR wrong file format, ongoing file reading is outdated/invalid
            return None;
        }
        match Array::from_shape_vec((depths, rows, columns), v_f64)
        {
            Ok(data) =>
            {
                if data.shape().iter().fold(true, |acc, &dim| acc & (dim > 1))
                {
                    return Some(TTInputData { data : data });
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
