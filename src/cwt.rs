use crate::tt_input_data::AtomicFileState;
use crate::tt_input_data::TTInputData;
use ndarray::Array3;
use std::sync::Arc;
use strum_macros::{EnumString, EnumVariantNames};

#[derive(
    Clone, Copy, PartialEq, Debug, Default, strum_macros::AsRefStr, EnumString, EnumVariantNames,
)]
#[strum(serialize_all = "title_case")]
pub enum WtResultMode
{
    #[default]
    Phase,
    Magnitude,
    Real,
    Imaginary,
}
#[derive(
    Clone, Copy, PartialEq, Debug, Default, strum_macros::AsRefStr, EnumString, EnumVariantNames,
)]
#[strum(serialize_all = "title_case")]
pub enum WaveletType
{
    #[default]
    Morlet,
}
pub struct TTInputIntegrated(pub Array3<f32>);

impl TTInputIntegrated
{
    pub fn new(input : &TTInputData, file_state : Arc<AtomicFileState>)
        -> Option<TTInputIntegrated>
    {
        None
    }
}
