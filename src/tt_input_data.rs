use atomic_enum::atomic_enum;
use ndarray::{Array, Array3};
use std::ffi::OsString;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::sync::atomic::Ordering;
use std::sync::Arc;
#[atomic_enum]
#[derive(PartialEq)]
pub enum FileState
{
    None,
    New,
    Loading,
    Loaded,
    Processing,
    Ready,
}

pub struct TTInputData(pub Array3<f32>);

impl Default for TTInputData
{
    fn default() -> Self { Self(Array3::default((1, 1, 1))) }
}

impl TTInputData
{
    pub fn new(path : OsString) -> Self
    {
        let f : File = match File::open(path.clone())
        {
            Ok(it) => it,
            Err(_) =>
            {
                return TTInputData::default();
            }
        };

        let fr = BufReader::new(f);
        // let mut file_content_string: String=String::new();
        // f.read_to_string(&mut file_content_string).unwrap();

        let mut parsing_header : bool = false;
        let mut columns : usize = 0;
        let mut rows : usize = 0;
        let mut depths : usize = 0;
        let mut v_f32 : Vec<f32> = Vec::default();

        for line in fr.lines()
        // for iline in file_content_string.lines()
        {
            match line
            {
                Ok(iline) =>
                {
                    if !iline.is_empty()
                    {
                        let mut v : Vec<&str> = iline.split_whitespace().collect();

                        if let Ok(_) = v[0].replace(',', ".").parse::<f32>()
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
                                v_f32.append(
                                    &mut v
                                        .iter()
                                        .map(|x| x.replace(',', ".").parse::<f32>().unwrap())
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

        TTInputData(Array::from_shape_vec((depths, rows, columns), v_f32).unwrap())
    }
}
