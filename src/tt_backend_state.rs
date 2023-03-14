use egui::{ColorImage, TextureOptions};
use ndarray::{s, ArrayView2, Axis};
use ndarray_ndimage::{convolve, BorderMode};
use parking_lot::{Condvar, Mutex};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::f64::consts::{FRAC_1_PI, PI};
use std::ffi::OsString;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use triple_buffer as tribuf;
use triple_buffer::triple_buffer;

use crate::cwt::*;
use crate::tt_common::*;
use crate::tt_input_data::*;
use crate::wavelet::{WaveletBank, WaveletBankTrait};

//=======================================
//=================Types=================
//=======================================
pub struct TTViewBackend
{
    pub state :      Arc<AtomicTTViewState>,
    pub thermogram : tribuf::Input<Thermogram>,
    pub params :     tribuf::Output<TTViewParams>,
    pub roi :        Arc<SemiAtomicRect>,
}
struct TTFileBackend
{
    frames :     Arc<AtomicUsize>,
    state :      Arc<AtomicFileState>,
    path :       tribuf::Output<Option<OsString>>,
    roi :        Arc<SemiAtomicRect>,
    input_data : Option<TTInputData>,
    lazy_cwt :   Option<TTLazyCWT>,
}

pub struct TTStateBackend
{
    views :        [TTViewBackend; 4],
    changed :      Arc<(Mutex<bool>, Condvar)>,
    stop_flag :    Arc<AtomicBool>,
    file :         TTFileBackend,
    wavelet_bank : WaveletBank,
}

//=======================================
//============Implementations============
//=======================================
impl TTViewBackend
{
    fn update_image<'a>(
        &mut self,
        array : ArrayView2<'a, f64>,
        grad : TTGradients,
        denoise : bool,
    ) -> ()
    {
        let array_base;
        let array = if denoise
        {
            //low pass filtring
            let filter =
                ndarray::arr2(&[[0.05_f64, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]]);
            array_base = convolve(&array.reborrow(), &filter.view(), BorderMode::Reflect, 0);
            array_base.view()
        }
        else
        {
            array.reborrow()
        };
        let array_roi = array.slice(s![
            self.roi.min.get().0 as usize..=self.roi.max.get().0 as usize,
            self.roi.min.get().1 as usize..=self.roi.max.get().1 as usize
        ]);
        let roi_len = array_roi.len();
        let non_roi_len = array.len() - roi_len;
        //replicate roi pixels `roi_mul`-1 times to ensure that roi pixels occurs(in histogram) as if roi spans above at least 70% of image
        let roi_mul = (7.0 / 3.0 * non_roi_len as f64 / roi_len as f64).ceil() as usize;
        let array_iter = array.into_par_iter().cloned();

        let mut array_vec = Vec::with_capacity(non_roi_len + roi_len * roi_mul + 2);
        array_vec.extend_from_slice(array.as_slice_memory_order().unwrap());
        let array_roi_iter = array_roi.iter();
        for _ in 1..roi_mul
        {
            array_vec.extend(array_roi_iter.clone());
        }
        if grad == TTGradients::Phase
        {
            //for phase gradient force -PI & PI as extreme vals
            array_vec.push(-PI);
            array_vec.push(PI);
        }
        array_vec
            .as_parallel_slice_mut()
            .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let quantile_count = (array_vec.len() - 1) as f64 / 32.0;
        let gram = self.thermogram.input_buffer();
        gram.scale.iter_mut().enumerate().for_each(|(idx, val)| {
            *val = array_vec[(quantile_count * idx as f64).round() as usize]
        });
        let quantile_out_width = 1.0 / 33.0;
        let mul_add_coef : Vec<(_, _)> = gram
            .scale
            .windows(2)
            .zip((0..32).map(|x| {
                (
                    x as f64 * quantile_out_width,
                    (x + 1) as f64 * quantile_out_width,
                )
            }))
            .map(|(x, (ymin, ymax))| {
                let (xmin, xmax) = (x[0], x[1]);
                let mul = (ymax - ymin) / (xmax - xmin);
                let add = ymin - xmin * mul;
                (mul, add)
            })
            .collect();
        let rgb = array_iter
            .map(|x| {
                let color;
                match gram.scale.binary_search_by(|a| a.partial_cmp(&x).unwrap())
                {
                    Ok(idx) =>
                    {
                        color = grad
                            .raw_grad()
                            .at(quantile_out_width * idx as f64)
                            .to_rgba8();
                    }
                    Err(idx) =>
                    {
                        //idx will always be in range <1;32>
                        let (mul, add) = mul_add_coef[idx - 1];
                        color = grad.raw_grad().at(f64::mul_add(x, mul, add)).to_rgba8();
                    }
                };
                [color[0], color[1], color[2]]
            })
            .flatten()
            .collect::<Vec<u8>>();
        let color_image = ColorImage::from_rgb([array.dim().1, array.dim().0], &rgb);
        gram.image.set(color_image, TextureOptions::LINEAR);
        if grad == TTGradients::Phase
        {
            //for phase scale values to degrees
            gram.scale
                .iter_mut()
                .for_each(|x| *x = *x * FRAC_1_PI * 180.0 + 180.0);
        }
        gram.legend = grad;
        self.thermogram.publish();
        let _ = self.state.compare_exchange(
            TTViewState::Processing,
            TTViewState::Valid,
            Ordering::SeqCst,
            Ordering::Acquire,
        );
    }
}
impl TTFileBackend
{
    pub fn new(
        frames : Arc<AtomicUsize>,
        state : Arc<AtomicFileState>,
        path : tribuf::Output<Option<OsString>>,
        roi : Arc<SemiAtomicRect>,
    ) -> Self
    {
        Self {
            frames,
            state,
            path,
            roi,
            input_data : None,
            lazy_cwt : None,
        }
    }
}
impl TTStateBackend
{
    pub fn new(
        views : [TTViewBackend; 4],
        changed : Arc<(Mutex<bool>, Condvar)>,
        stop_flag : Arc<AtomicBool>,
        frames : Arc<AtomicUsize>,
        state : Arc<AtomicFileState>,
        path : tribuf::Output<Option<OsString>>,
        roi : Arc<SemiAtomicRect>,
    ) -> Self
    {
        Self {
            views,
            changed,
            stop_flag,
            file : TTFileBackend::new(frames, state, path, roi),
            wavelet_bank : WaveletBank::new_wb(),
        }
    }

    fn check_changed_and_sleep(&mut self) -> ()
    {
        let &(ref mx, ref cvar) = &*self.changed;
        let mut mxval = mx.lock();
        if *mxval == false
        {
            //no new changes backend can sleep
            cvar.wait(&mut mxval);
        }
        *mxval = false;
    }
    pub fn run(mut self) -> ()
    {
        while self.stop_flag.load(Ordering::Relaxed) == false
        {
            match self.file.state.load(Ordering::Relaxed)
            {
                FileState::None =>
                {
                    self.file.input_data = None;
                    self.file.lazy_cwt = None;
                    self.check_changed_and_sleep();
                }
                FileState::New =>
                {
                    //input file path changed
                    let _ = self.file.state.compare_exchange(
                        FileState::New,
                        FileState::Loading,
                        Ordering::SeqCst,
                        Ordering::Acquire,
                    );
                }
                FileState::Loading =>
                {
                    self.file.lazy_cwt = None;
                    if let Some(path) = self.file.path.read()
                    {
                        self.file.input_data = TTInputData::new(path, self.file.state.clone());
                    }
                    else
                    {
                        unreachable!()
                    }
                    if let Some(input) = &self.file.input_data
                    {
                        //file loaded correctly
                        self.file
                            .frames
                            .store(input.data.len_of(ndarray::Axis(0)), Ordering::Relaxed);
                        let (x, y) = (input.data.shape()[1] as u16, input.data.shape()[2] as u16);
                        let min = (x / 8, y / 8);
                        self.file.roi.min.set(min);
                        self.file.roi.max.set((min.0 * 7, min.1 * 7));
                        self.file.roi.full_size.set((x, y));

                        let _ = self.file.state.compare_exchange(
                            FileState::Loading,
                            FileState::Loaded,
                            Ordering::SeqCst,
                            Ordering::Acquire,
                        );
                    }
                }
                FileState::Loaded =>
                    //this state is only for gui to acknowlege processing completed
                    {}
                FileState::Processing =>
                {
                    if let Some(input) = &self.file.input_data
                    {
                        self.file.lazy_cwt = None;
                        rayon::join(
                            || {
                                //continously update time views if necessary
                                while FileState::Processing
                                    == self.file.state.load(Ordering::Relaxed)
                                    && self.stop_flag.load(Ordering::Relaxed) == false
                                {
                                    for view in &mut self.views
                                    {
                                        if let TimeView(params) = view.params.read()
                                        {
                                            if let Ok(_) = view.state.compare_exchange(
                                                TTViewState::Changed,
                                                TTViewState::Processing,
                                                Ordering::SeqCst,
                                                Ordering::Acquire,
                                            )
                                            {
                                                if let Some(input) = &self.file.input_data
                                                {
                                                    let input_view = input
                                                        .data
                                                        .index_axis(Axis(0), params.time.val);
                                                    let denoise = params.denoise;
                                                    view.update_image(
                                                        input_view,
                                                        TTGradients::Linear,
                                                        denoise,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            || {
                                self.file.lazy_cwt =
                                    TTLazyCWT::new(&input.data, self.file.state.clone());

                                if let Some(_) = &self.file.lazy_cwt
                                {
                                    //file processed correctly
                                    let _ = self.file.state.compare_exchange(
                                        FileState::Processing,
                                        FileState::Ready,
                                        Ordering::SeqCst,
                                        Ordering::Acquire,
                                    );
                                }
                            },
                        );
                    }
                    else
                    {
                        unreachable!()
                    }
                }
                FileState::Ready =>
                {
                    for view in &mut self.views
                    {
                        if let Ok(_) = view.state.compare_exchange(
                            TTViewState::Changed,
                            TTViewState::Processing,
                            Ordering::SeqCst,
                            Ordering::Acquire,
                        )
                        {
                            match view.params.read()
                            {
                                TimeView(params) =>
                                {
                                    if let Some(input) = &self.file.input_data
                                    {
                                        let input_view =
                                            input.data.index_axis(Axis(0), params.time.val);
                                        let denoise = params.denoise;
                                        view.update_image(input_view, TTGradients::Linear, denoise);
                                    }
                                }
                                TransformView(params) =>
                                {
                                    if let Some(cwt) = &self.file.lazy_cwt
                                    {
                                        //calculate & display requested waveletet transform
                                        let cwt_view = cwt.cwt(&mut self.wavelet_bank, params);
                                        let denoise = params.denoise;
                                        if params.mode == WtResultMode::Phase
                                        {
                                            view.update_image(
                                                cwt_view.view(),
                                                TTGradients::Phase,
                                                denoise,
                                            );
                                        }
                                        else
                                        {
                                            view.update_image(
                                                cwt_view.view(),
                                                TTGradients::Linear,
                                                denoise,
                                            );
                                        };
                                    }
                                }
                            }
                        }
                    }
                    self.check_changed_and_sleep();
                }
            }
        }
    }
}
