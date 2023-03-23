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
        let f : File = match File::open(path.clone())
        {
            Ok(it) => it,
            Err(_) =>
            {
                file_state.store(FileState::Error, Ordering::SeqCst);
                return None;
            }
        };

        let fr = BufReader::new(f);

        let mut parsing_header : bool = false;
        let mut columns : usize = 0;
        let mut rows : usize = 0;
        let mut depths : usize = 0;
        let mut v_f64 : Vec<f64> = Vec::default();

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
                        let mut v : Vec<&str> = iline.split_whitespace().collect();

                        if let Ok(_) = v[0].replace(',', ".").parse::<f64>()
                        {
                            //if first string in linie is number
                            if parsing_header
                            {
                                //first linie with numbers are column id's; skip it
                                parsing_header = false;
                                columns = 0;
                                rows = 0;
                                depths += 1;
                            }
                            else
                            {
                                v.remove(0); //first number is row id
                                columns = v.len();
                                rows += 1;
                                v_f64.append(
                                    &mut v
                                        .iter()
                                        .map(|x| {
                                            let parse_result = x.replace(',', ".").parse::<f64>();
                                            let x = match parse_result
                                            {
                                                Ok(val) => val,
                                                Err(_) =>
                                                {
                                                    //structure of input file is not as expected(wrong file format), function finishes current line and returns faliure
                                                    file_state
                                                        .store(FileState::Error, Ordering::SeqCst);
                                                    f64::NAN
                                                }
                                            };
                                            x
                                        })
                                        .collect(),
                                );
                            }
                        }
                        else
                        {
                            //line belongs to header/subheader; skip it
                            parsing_header = true;
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
