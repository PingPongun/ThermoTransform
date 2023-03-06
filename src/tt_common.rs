use atomic_enum::atomic_enum;
use std::str::FromStr;
use strum::VariantNames;
use strum_macros::EnumVariantNames;

use crate::cwt::*;
use crate::wavelet::WaveletType;

//=======================================
//=================Types=================
//=======================================

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
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct TransformViewParams
{
    pub scale :   RangedVal,
    pub time :    RangedVal,
    pub wavelet : WaveletType,
    pub mode :    WtResultMode,
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
    TransformView(TransformViewParams),
    TimeView(RangedVal),
}

impl TTViewParams
{
    pub fn time_default() -> Self { TimeView(Default::default()) }
    pub fn transform_default() -> Self { TransformView(Default::default()) }
    pub fn time_frames(frames : usize) -> Self
    {
        TimeView(RangedVal {
            val : 0,
            min : 0,
            max : frames - 1,
        })
    }
    pub fn transform_frames(frames : usize) -> Self
    {
        TransformView(TransformViewParams {
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
        })
    }
    pub fn transform_wavelet(wavelet : WaveletType,mode:WtResultMode) -> Self
    {
        TransformView(TransformViewParams {
            time :    Default::default(),
            scale :   Default::default(),
            wavelet ,
            mode ,
        })
    }
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
        let timeview : &'static str = Self::time_default().into();
        let transformview : &'static str = Self::transform_default().into();
        let frames;
        match self
        {
            TimeView(time) => frames = time.max + 1,
            TransformView(params) => frames = params.scale.max,
        };
        if timeview == new_val
        {
            *self = Self::time_frames(frames)
        }
        else if transformview == new_val
        {
            *self = Self::transform_frames(frames)
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
