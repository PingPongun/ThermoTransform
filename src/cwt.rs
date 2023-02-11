use crate::tt_input_data::AtomicFileState;
use std::sync::Arc;

use ndarray::Array3;

use crate::tt_input_data::TTInputData;
pub struct TTInputIntegrated(pub Array3<f32>);

impl TTInputIntegrated
{
    pub fn new(input : &TTInputData, file_state : Arc<AtomicFileState>)
        -> Option<TTInputIntegrated>
    {
        None
    }
}
