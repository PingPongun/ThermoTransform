use ndarray::{Array, Array3};
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use std::ffi::OsString;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::tt_common::*;
#[derive(PartialEq)]
pub struct TTInputData
{
    pub data :    Array3<f64>,
    pub min_val : f64,
    pub max_val : f64,
}

impl TTInputData
{
    pub fn new(path : &OsString, file_state : Arc<AtomicFileState>) -> Option<Self>
    {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        let f : File = match File::open(path.clone())
        {
            Ok(it) => it,
            Err(_) =>
            {
                return None;
            }
        };

        let fr = BufReader::new(f);
        // let mut file_content_string: String=String::new();
        // f.read_to_string(&mut file_content_string).unwrap();

        let mut parsing_header : bool = false;
        let mut columns : usize = 0;
        let mut rows : usize = 0;
        let mut depths : usize = 0;
        let mut v_f64 : Vec<f64> = Vec::default();

        for line in fr.lines()
        // for iline in file_content_string.lines()
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
                                            let x = x.replace(',', ".").parse::<f64>().unwrap();
                                            min_val = min_val.min(x);
                                            max_val = max_val.max(x);
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
        let mul = 1.0 / (max_val - min_val);
        let add = -min_val * mul;
        v_f64 = v_f64.into_par_iter().map(|x| (x * mul + add)).collect();
        Some(TTInputData {
            data : Array::from_shape_vec((depths, rows, columns), v_f64).unwrap(),
            min_val,
            max_val,
        })
    }
}
