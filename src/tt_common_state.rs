use atomic_enum::atomic_enum;
use std::str::FromStr;
use strum::VariantNames;
use strum_macros::EnumVariantNames;

use crate::cwt::*;

//=======================================
//=================Types=================
//=======================================

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RangedVal
{
    pub val : usize,
    pub min : usize,
    pub max : usize,
}

#[atomic_enum]
#[derive(PartialEq)]
pub enum TTViewState
{
    Valid,
    Processing,
    Changed,
    Invalid,
}

#[derive(
    Clone,
    Copy,
    PartialEq,
    Debug,
    strum_macros::IntoStaticStr,
    strum_macros::AsRefStr,
    EnumVariantNames,
)]
#[strum(serialize_all = "title_case")]
pub enum TTViewParams
{
    TransformView
    {
        scale :   RangedVal,
        time :    RangedVal,
        wavelet : WaveletType,
        mode :    WtResultMode,
    },
    TimeView
    {
        time : RangedVal
    },
}
pub use TTViewParams::{TimeView, TransformView};

//=======================================
//=====Traits & Trait Implementations====
//=======================================

pub trait EnumUpdate<T>
{
    fn update(&mut self, new_val : &str);
}

impl<T> EnumUpdate<T> for T
where
    T : VariantNames,
    T : AsRef<str>,
    T : FromStr,
    <T as FromStr>::Err : std::fmt::Debug,
{
    fn update(&mut self, new_val : &str) { *self = T::from_str(new_val).unwrap(); }
}
impl EnumUpdate<TTViewParams> for TTViewParams
{
    fn update(&mut self, new_val : &str)
    {
        let timeview : &'static str = (TimeView {
            time : Default::default(),
        })
        .into();
        let transformview : &'static str = (TransformView {
            time :    Default::default(),
            scale :   Default::default(),
            wavelet : Default::default(),
            mode :    Default::default(),
        })
        .into();
        let frames;
        match self
        {
            TimeView { time } => frames = time.max + 1,
            TransformView {
                scale,
                time: _,
                wavelet: _,
                mode: _,
            } => frames = scale.max,
        };
        if timeview == new_val
        {
            *self = TimeView {
                time : RangedVal {
                    val : 0,
                    min : 0,
                    max : frames - 1,
                },
            }
        }
        else if transformview == new_val
        {
            *self = TransformView {
                time :    RangedVal {
                    val : 0,
                    min : 0,
                    max : frames - 1,
                },
                scale :   RangedVal {
                    val : 1,
                    min : 1,
                    max : frames,
                },
                wavelet : Default::default(),
                mode :    Default::default(),
            }
        }
        else
        {
            unreachable!()
        }
    }
}

//=======================================
//============Implementations============
//=======================================

impl Default for RangedVal
{
    fn default() -> Self
    {
        Self {
            val : 0,
            min : 0,
            max : 100,
        }
    }
}
