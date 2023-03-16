use ndarray::Array2;
use ndarray::Array3;
use ndarray::Axis;
use ndrustfft::ndfft_r2c_par;
use ndrustfft::R2cFftHandler;
use num_complex::Complex64;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::tt_common::*;

pub struct TTFourier
{
    data : Array3<Complex64>,
}
impl TTFourier
{
    pub fn new(input : &Array3<f64>, file_state : Arc<AtomicFileState>) -> Option<TTFourier>
    {
        let mut shape_raw = input.dim();
        let mut fft_handler = R2cFftHandler::<f64>::new(shape_raw.0);
        shape_raw.0 = shape_raw.0 / 2 + 1;
        let mut fourier = TTFourier {
            data : Array3::default(shape_raw),
        };
        ndfft_r2c_par(input, &mut fourier.data, &mut fft_handler, 0);

        if FileState::ProcessingFourier == file_state.load(Ordering::Relaxed)
        {
            Some(fourier)
        }
        else
        {
            None
        }
    }

    pub fn snapshot(&self, params : &FourierViewParams) -> Array2<f64>
    {
        let freq_view = self.data.index_axis(Axis(0), params.freq.val);
        //convert to requested format
        match params.display_mode
        {
            ComplexResultMode::Phase => freq_view.map(|x| x.arg()), //radians
            ComplexResultMode::Magnitude => freq_view.map(|x| x.norm()),
            ComplexResultMode::Real => freq_view.split_complex().re.to_owned(),
            ComplexResultMode::Imaginary => freq_view.split_complex().im.to_owned(),
        }
    }
}
