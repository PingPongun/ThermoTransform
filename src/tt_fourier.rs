use ndarray::s;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::Axis;
use ndarray::Slice;
use ndrustfft::ndfft_r2c_par;
use ndrustfft::ndifft_r2c_par;
use ndrustfft::R2cFftHandler;
use num_complex::Complex;
use num_complex::Complex64;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::IntoParallelRefMutIterator;
use rayon::prelude::ParallelBridge;
use rayon::prelude::ParallelIterator;
use std::f64::consts::PI;
use std::mem;
use std::mem::MaybeUninit;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::gap_window::GAPWin;
use crate::tt_common::*;
pub struct TTFourier
{
    time_len :            usize,
    time_len_wo_padding : usize,
    data :                Array3<Complex64>,
}
#[derive(Clone)]
struct TTFourierUninit
{
    _time_len :            usize,
    _time_len_wo_padding : usize,
    data :                 Array3<MaybeUninit<Complex<f64>>>,
}

impl TTFourierUninit
{
    fn new<const N: usize>(
        shape : (usize, usize, usize),
        time_len : usize,
        time_len_wo_padding : usize,
    ) -> [TTFourierUninit; N]
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
                data :                 Array3::uninit(shape),
                _time_len :            time_len,
                _time_len_wo_padding : time_len_wo_padding,
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
    pub fn new(input : &TTInputData, file_state : Arc<AtomicFileState>) -> Option<TTFourier>
    {
        let mut shape_raw = input.data.dim();
        let mut fft_handler = R2cFftHandler::<f64>::new(shape_raw.2);
        let time_len = shape_raw.2;
        shape_raw.2 = shape_raw.2 / 2 + 1;
        let mut fourier = TTFourier {
            data : Array3::zeros(shape_raw),
            time_len,
            time_len_wo_padding : input.frames,
        };
        let mut windowed_data = input.data.to_owned();
        let window = GAPWin::KAISER_17_OPT.window(input.frames);
        windowed_data
            .lanes_mut(AXIS_T)
            .into_iter()
            .into_par_iter()
            .for_each(|mut lane| lane.iter_mut().zip(&window).for_each(|(i, w)| *i *= w));
        ndfft_r2c_par(&windowed_data, &mut fourier.data, &mut fft_handler, 2);

        if FileState::ProcessingFourier == file_state.load(Ordering::Relaxed)
        {
            Some(fourier)
        }
        else
        {
            None
        }
    }

    pub fn snapshot(&self, params : &ViewMode, settings : &GlobalSettings) -> Array2<f64>
    {
        let settings_axis = params.get_settings_axes()[0] as usize;
        let freq_view = self.data.index_axis(
            Axis(
                if settings_axis == TTAxis::F as usize
                {
                    2
                }
                else
                {
                    settings_axis
                },
            ),
            params.position.read()[settings_axis],
        );
        let freq_view = if settings.roi_zoom.load(Ordering::Relaxed)
        {
            let view_axes = params.get_view_axes();
            let roi_h = settings.get_roi(view_axes[0]);
            let roi_v = settings.get_roi(view_axes[1]);
            freq_view.slice(s![roi_h, roi_v])
        }
        else
        {
            freq_view
        };
        //convert to requested format
        match params.display_mode.load(Ordering::Relaxed)
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
        // X_k*(-w*i)           ->  w1*X_k_i - i*w1*X_k_r
        // X_k*(-w*i)^2         -> -w2*X_k_r - i*w2*X_k_i
        // X_k*(-w*i)^3         -> -w3*X_k_i + i*w3*X_k_r
        // X_k*(-w*i)^4         ->  w4*X_k_r + i*w4*X_k_i
        // let mut exec_time = ExecutionTimeMeas::new("exec_time_fourier_init.txt");
        // exec_time.start();
        let shape = self.data.dim();

        let mut uninit_data : [TTFourierUninit; N] =
            TTFourierUninit::new(shape, self.time_len, self.time_len_wo_padding);
        // exec_time.stop_print("uninit");
        // exec_time.start();

        let w_coeff_base = (0..shape.2)
            .map(|x| 1.0 / (2.0 * x as f64 * PI))
            .collect::<Vec<_>>();
        let w_coeffs_odd = (1..=N)
            .into_iter()
            .step_by(2)
            .map(|idx| {
                w_coeff_base
                    .iter()
                    .map(move |w| {
                        //* ((1 - (idx & 0x2) as isize) as f64) == -1 when idx= 3,7,11,.. ; == 1 when idx= 1,5,9,..
                        let w = w.powi(idx as i32) * ((1 - (idx & 0x2) as isize) as f64);
                        (w, -w)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let w_coeffs_even = (2..=N)
            .into_iter()
            .step_by(2)
            .map(|idx| {
                w_coeff_base
                    .iter()
                    .map(move |w| {
                        //* ((1 - (idx & 0x2) as isize) as f64) == -1 when idx= 2,6,10,.. ; == 1 when idx= 4,8,12,..
                        let w = w.powi(idx as i32) * ((1 - (idx & 0x2) as isize) as f64);
                        w
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // exec_time.stop_print("coefs");
        // exec_time.start();

        //initialize integrals 1st, 3rd, 5th,..
        uninit_data
            .iter_mut()
            .step_by(2)
            .zip(w_coeffs_odd)
            .par_bridge()
            .for_each(|(x, w_coeffs)| {
                x.data
                    .lanes_mut(AXIS_T)
                    .into_iter()
                    .into_par_iter()
                    .zip(self.data.lanes(AXIS_T).into_iter())
                    .for_each(|(mut oarray, iarray)| {
                        oarray
                            .as_slice_memory_order_mut()
                            .unwrap()
                            .iter_mut()
                            .zip(iarray.as_slice_memory_order().unwrap())
                            .zip(&w_coeffs)
                            .for_each(|((o, i), w)| {
                                o.write(Complex64 {
                                    re : w.0 * i.im,
                                    im : w.1 * i.re,
                                });
                            })
                    })
            });
        // exec_time.stop_print("integrals 1");
        // exec_time.start();

        //initialize integrals 2nd, 4th, 6th,..
        uninit_data
            .iter_mut()
            .skip(1)
            .step_by(2)
            .zip(w_coeffs_even)
            .par_bridge()
            .for_each(|(x, w_coeffs)| {
                x.data
                    .lanes_mut(AXIS_T)
                    .into_iter()
                    .into_par_iter()
                    .zip(self.data.lanes(AXIS_T).into_iter())
                    .for_each(|(mut oarray, iarray)| {
                        oarray
                            .as_slice_memory_order_mut()
                            .unwrap()
                            .iter_mut()
                            .zip(iarray.as_slice_memory_order().unwrap())
                            .zip(&w_coeffs)
                            .for_each(|((o, i), w)| {
                                o.write(Complex64 {
                                    re : w * i.re,
                                    im : w * i.im,
                                });
                            })
                    })
            });
        // exec_time.stop_print("integrals 2");
        // exec_time.start();

        // Everything is initialized. Transmute the array to the
        // initialized type.
        let ptr = &mut uninit_data as *mut _ as *mut [TTFourier; N];
        let mut res = unsafe { ptr.read() };
        mem::forget(uninit_data);
        // exec_time.stop_print("cast");
        // exec_time.start();

        res.par_iter_mut().for_each(|x| {
            let fill = Complex64::new(0.0, 0.0);
            x.data.index_axis_mut(AXIS_T, 0).fill(fill);
        });
        // exec_time.stop_print("end");

        res
    }

    pub fn inverse_transform(&self) -> Array3<f64>
    {
        let mut shape = self.data.dim();
        shape.2 = self.time_len;
        let mut handler = R2cFftHandler::<f64>::new(shape.2);
        let mut output = Array3::zeros(shape);
        ndifft_r2c_par(&self.data, &mut output, &mut handler, 2);
        output.slice_axis_inplace(AXIS_T, Slice::from(..self.time_len_wo_padding));
        output
    }

    pub fn integrals<const N: usize>(&self) -> [Array3<f64>; N]
    {
        // let mut exec_time = ExecutionTimeMeas::new("exec_time_fourier.txt");
        // exec_time.start();
        let temp = self.integrals_dft();
        // exec_time.stop_print("initial");
        // exec_time.start();
        let mut ret = temp.map(|x| x.inverse_transform());
        // exec_time.stop_print("ifft");
        //remove window from signal
        // this is not fully correct!!!, as integral is not (fi*w).(fi*x), but fi*(w.x), where fi is operation taken in Fourier domain to integrate and '.' is multiplication, w is window, x is signal
        ret.par_iter_mut().enumerate().for_each(|(i, integral3d)| {
            let win = GAPWin::KAISER_17_OPT.integrated_window(self.time_len_wo_padding, i);
            let iwin = win.map(|w| 1. / w);
            integral3d
                .lanes_mut(AXIS_T)
                .into_iter()
                .into_par_iter()
                .for_each(|mut lane| lane.iter_mut().zip(&iwin).for_each(|(i, w)| *i *= w))
        });

        // exec_time.stop_print("de-window");
        ret
    }
}
