use ndarray::Array2;
use ndarray::Array3;
use ndarray::Axis;
use ndrustfft::ndfft_r2c_par;
use ndrustfft::ndifft_r2c_par;
use ndrustfft::R2cFftHandler;
use num_complex::Complex;
use num_complex::Complex64;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use std::f64::consts::PI;
use std::mem;
use std::mem::MaybeUninit;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::tt_common::*;

pub struct TTFourier
{
    data : Array3<Complex64>,
}
#[derive(Clone)]
struct TTFourierUninit
{
    data : Array3<Complex<MaybeUninit<f64>>>,
}

impl TTFourierUninit
{
    fn new<const N: usize>(shape : (usize, usize, usize)) -> [TTFourierUninit; N]
    {
        // Create an uninitialized array of `MaybeUninit`. The `assume_init` is
        // safe because the type we are claiming to have initialized here is a
        // bunch of `MaybeUninit`s, which do not require initialization.
        let mut uninit_data : [MaybeUninit<TTFourierUninit>; N] =
            unsafe { MaybeUninit::uninit().assume_init() };

        // Dropping a `MaybeUninit` does nothing, so if there is a panic during this loop,
        // we have a memory leak, but there is no memory safety issue.

        for elem in &mut uninit_data[..]
        {
            elem.write(TTFourierUninit {
                data : Array3::uninit(shape).map(|x| unsafe { x.assume_init() }),
            });
        }

        // Everything is initialized. Transmute the array to the
        // initialized type.
        let ptr = &mut uninit_data as *mut _ as *mut [TTFourierUninit; N];
        let res = unsafe { ptr.read() };
        mem::forget(uninit_data);
        res
    }
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

    fn integrals_dft<const N: usize>(&self) -> [TTFourier; N]
    {
        // each component of DFT (X_k) encodes ampltude and phase of cos(2*PI*f_k*t)+i*sin(2*PI*f_k*t) [==e^(2*i*PI*f_k*t)]
        // where f_k = k/N & k-> DFT component index & N-> Number of DFT components
        // integration of function that can be represented through fourier series (as SUM(X_k * e^(2*i*PI*f_k*t)) )
        // is equal SUM(X_k * e^(2*i*PI*f_k*t) * 1/(2*i*PI*f_k) )
        // which can be calculated through IDFT of (DFT components(X_k) multiplied by 1/(2*i*PI*f_k) )
        // 1/(2*i*PI*f_k) -> w/i where w=1/(2*PI*f_k)
        // w/i-> -w*i
        // X_k= X_k_r + i*X_k_i
        // X_k*(-w*i)           -> w*X_k_i - i*w*X_k_r
        let shape = self.data.dim();

        let mut uninit_data : [TTFourierUninit; N] = TTFourierUninit::new(shape);
        let w_coeff = (0..shape.0)
            .map(|x| {
                let w = 1.0 / (2.0 * x as f64 * PI);
                (w, -w)
            })
            .collect::<Vec<_>>();
        uninit_data[0]
            .data
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .zip(self.data.axis_iter(Axis(0)))
            .zip(w_coeff.clone())
            .for_each(|((mut oarray, iarray), w)| {
                oarray.zip_mut_with(&iarray, |o, i| {
                    o.re = MaybeUninit::new(w.0 * i.im);
                    o.im = MaybeUninit::new(w.1 * i.re);
                })
            });
        let celled : Vec<_> = uninit_data.iter_mut().map(|x| x.data.cell_view()).collect();
        //TODO ? paralellization
        celled.windows(2).for_each(|x| {
            x[1].axis_iter(Axis(0))
                .zip(x[0].axis_iter(Axis(0)))
                .zip(w_coeff.clone())
                .for_each(|((oarray, iarray), w)| {
                    oarray.iter().zip(iarray.iter()).for_each(|(o, i)| {
                        let i = unsafe { (i.as_ptr() as *const Complex64).read() }; //it's safe becouse i~iarray~x[0] was initialized in previous step
                        let o = o.as_ptr();
                        unsafe {
                            (*o).re = MaybeUninit::new(w.0 * i.im);
                            (*o).im = MaybeUninit::new(w.1 * i.re);
                        }
                    })
                });
        });
        // Everything is initialized. Transmute the array to the
        // initialized type.
        let ptr = &mut uninit_data as *mut _ as *mut [TTFourier; N];
        let mut res = unsafe { ptr.read() };
        res.iter_mut().for_each(|x| {
            let fill = Complex64::new(0.0, 0.0);
            x.data.index_axis_mut(Axis(0), 0).fill(fill);
        });
        mem::forget(uninit_data);
        res
    }

    pub fn inverse_transform(&self) -> Array3<f64>
    {
        let mut shape = self.data.dim();
        shape.0 = (shape.0 - 1) * 2;
        let mut handler = R2cFftHandler::<f64>::new(shape.0);
        let mut output = Array3::default(shape);
        ndifft_r2c_par(&self.data, &mut output, &mut handler, 0);
        output
    }

    pub fn integrals<const N: usize>(&self) -> [Array3<f64>; N]
    {
        self.integrals_dft().map(|x| x.inverse_transform())
    }
}
