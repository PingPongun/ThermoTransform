use egui::{ColorImage, TextureHandle, TextureOptions};
use ndarray::{ArrayView2, Axis};
use parking_lot::{Condvar, Mutex};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::ffi::OsString;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use triple_buffer as tribuf;
use triple_buffer::triple_buffer;

use crate::cwt::*;
use crate::tt_common::*;
use crate::tt_input_data::*;
use crate::wavelet::WaveletBank;

//=======================================
//=================Types=================
//=======================================
pub struct TTViewBackend
{
    pub state :  Arc<AtomicTTViewState>,
    pub image :  tribuf::Input<TextureHandle>,
    pub params : tribuf::Output<TTViewParams>,
}
pub struct TTFileBackend
{
    pub frames :           Arc<AtomicUsize>,
    pub state :            Arc<AtomicFileState>,
    pub path :             tribuf::Output<Option<OsString>>,
    pub input_data :       Option<TTInputData>,
    pub input_integrated : Option<TTLazyCWT>,
}
pub struct TTStateBackend
{
    pub views :        [TTViewBackend; 4],
    pub changed :      Arc<(Mutex<bool>, Condvar)>,
    pub stop_flag :    Arc<AtomicBool>,
    pub file :         TTFileBackend,
    pub wavelet_bank : WaveletBank,
}
//=======================================
//============Implementations============
//=======================================
impl TTViewBackend
{
    fn update_image<'a>(&mut self, array : ArrayView2<'a, f64>, max_local_contrast : bool) -> ()
    {
        let grad = colorgrad::inferno();
        let array_iter = array.into_par_iter().cloned();
        let rgb;
        if max_local_contrast
        {
            // version providing maximal contrast
            // find minimal and maximal value of pixel"
            let (min, max) = array_iter
                .clone()
                .fold(
                    || (f64::INFINITY, f64::NEG_INFINITY),
                    |(min, max), x| (min.min(x), max.max(x)),
                )
                .reduce(
                    || (f64::INFINITY, f64::NEG_INFINITY),
                    |(min, max), (xmin, xmax)| (min.min(xmin), max.max(xmax)),
                );
            let mul = 1.0 / (max - min);
            let add = -min * mul;
            rgb = array_iter
                .map(|x| {
                    let color = grad.at((x * mul + add).into()).to_rgba8(); //scale value to 0..=1 range
                    [color[0], color[1], color[2]]
                })
                .flatten()
                .collect::<Vec<u8>>();
        }
        else
        {
            rgb = array_iter
                .map(|x| {
                    let color = grad.at(x.into()).to_rgba8(); //scale value to 0..=1 range
                    [color[0], color[1], color[2]]
                })
                .flatten()
                .collect::<Vec<u8>>();
        }
        let color_image = ColorImage::from_rgb([array.dim().1, array.dim().0], &rgb);
        self.image
            .input_buffer()
            .set(color_image, TextureOptions::LINEAR);
        self.image.publish();
        let _ = self.state.compare_exchange(
            TTViewState::Processing,
            TTViewState::Valid,
            Ordering::SeqCst,
            Ordering::Acquire,
        );
    }
}
impl TTStateBackend
{
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
                    self.file.input_integrated = None;
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
                    self.file.input_integrated = None;
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
                        self.file.input_integrated = None;
                        rayon::join(
                            || {
                                //continously update time views if necessary
                                while FileState::Processing
                                    == self.file.state.load(Ordering::Relaxed)
                                    && self.stop_flag.load(Ordering::Relaxed) == false
                                {
                                    for view in &mut self.views
                                    {
                                        if let TimeView(time) = view.params.read()
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
                                                    let input_view =
                                                        input.data.index_axis(Axis(0), time.val);
                                                    view.update_image(input_view, false);
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            || {
                                self.file.input_integrated =
                                    TTLazyCWT::new(&input.data, self.file.state.clone());

                                if let Some(_) = &self.file.input_integrated
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
                                TimeView(time) =>
                                {
                                    if let Some(input) = &self.file.input_data
                                    {
                                        let input_view = input.data.index_axis(Axis(0), time.val);
                                        view.update_image(input_view, false);
                                    }
                                }
                                TransformView(params) =>
                                {
                                    if let Some(cwt) = &self.file.input_integrated
                                    {
                                        //calculate & display requested waveletet transform
                                        let cwt_view = cwt.cwt(&mut self.wavelet_bank, params);
                                        view.update_image(cwt_view.view(), true);
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
