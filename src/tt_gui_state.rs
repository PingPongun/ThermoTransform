use egui::{
    Color32,
    ColorImage,
    Context,
    DragValue,
    Image,
    Pos2,
    Rect,
    RichText,
    Sense,
    Spinner,
    Stroke,
    TextureOptions,
    Vec2,
};
use egui_extras::{Column, TableBuilder};
use parking_lot::{Condvar, Mutex};
use std::ffi::OsString;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use strum::VariantNames;
use triple_buffer as tribuf;
use triple_buffer::triple_buffer;

use crate::tt_backend_state::*;
use crate::tt_common::*;
use crate::wavelet::WaveletType;

//=======================================
//=================Types=================
//=======================================

pub struct TTViewGUI
{
    state :  Arc<AtomicTTViewState>,
    image :  tribuf::Output<Thermogram>,
    params : tribuf::Input<TTViewParams>,
    roi :    Arc<SemiAtomicRect>,
}

pub struct TTFileGUI
{
    frames : Arc<AtomicUsize>,
    state :  Arc<AtomicFileState>,
    path :   tribuf::Input<Option<OsString>>,
    roi :    Arc<SemiAtomicRect>,
}

pub struct TTStateGUI
{
    views :          [TTViewGUI; 4],
    changed :        Arc<(Mutex<bool>, Condvar)>,
    stop_flag :      Arc<AtomicBool>,
    file :           TTFileGUI,
    backend_handle : Option<JoinHandle<()>>,
}

//=======================================
//=====Traits & Trait Implementations====
//=======================================
trait BoolSwitchable<T>
{
    fn show_switchable(&mut self, ui : &mut egui::Ui, text : &str) -> bool;
}
impl BoolSwitchable<bool> for bool
{
    fn show_switchable(&mut self, ui : &mut egui::Ui, text : &str) -> bool
    {
        let mut retval = false;
        let text = if *self == true
        {
            RichText::new(text)
        }
        else
        {
            RichText::new(text).strikethrough()
        };
        if ui.button(text).clicked()
        {
            *self = !*self;
            retval = true;
        }
        retval
    }
}
trait EnumCombobox<T>
{
    fn show_combobox(&mut self, ui : &mut egui::Ui) -> bool;
}
impl<T> EnumCombobox<T> for T
where
    T : VariantNames,
    T : AsRef<str>,
    T : EnumUpdate<T>,
    T : Clone,
{
    fn show_combobox(&mut self, ui : &mut egui::Ui) -> bool
    {
        let mut retval = false;
        ui.spacing_mut().combo_width = 0.0;
        ui.spacing_mut().icon_spacing = 0.0;
        ui.spacing_mut().item_spacing = [5.0, 10.0].into();
        ui.spacing_mut().icon_width = 0.0;
        ui.spacing_mut().button_padding = [2.0, 2.0].into();
        ui.set_min_width(10.0);
        egui::ComboBox::from_id_source(ui.next_auto_id())
            .selected_text(self.as_ref())
            .icon(|_, _, _, _, _| {})
            .show_ui(ui, |ui| {
                let cloneself = self.clone();
                let mut current = cloneself.as_ref();
                ui.style_mut().wrap = Some(false);
                for &var in T::VARIANTS
                {
                    if ui.selectable_value(&mut current, var, var).changed()
                    {
                        retval = true;
                    }
                }
                if retval
                {
                    self.update(current);
                }
            });
        retval
    }
}
//=======================================
//=====Common Types Implementations======
//=======================================
impl RangedVal
{
    fn show(&mut self, ui : &mut egui::Ui) -> bool
    {
        ui.add(DragValue::new(&mut self.val).clamp_range(self.min..=self.max))
            .changed()
    }
}
impl Thermogram
{
    fn show(
        &self,
        ui : &mut egui::Ui,
        roi : &Arc<SemiAtomicRect>,
    ) -> (bool, egui::InnerResponse<()>)
    {
        let available_size = ui.available_size();
        let image_aspect = self.image.aspect_ratio();
        let mut retval = false;
        let size = if (available_size.x / available_size.y) > image_aspect
        {
            //available space is proportionaly wider than original image
            Vec2 {
                x : available_size.y * image_aspect,
                y : available_size.y,
            }
        }
        else
        {
            //available space is proportionaly taller than original image
            Vec2 {
                x : available_size.x,
                y : available_size.x / image_aspect,
            }
        };
        let responce = ui.horizontal_centered(|ui| {
            let img_rsp = ui.add(Image::new(self.image.id(), size).sense(Sense::click()));
            let size = img_rsp.rect.size();
            let roi_min = roi.min.get();
            let roi_max = roi.max.get();
            let full_size = roi.full_size.get();
            if img_rsp.clicked()
            {
                //left click
                let click_pos = img_rsp.interact_pointer_pos().unwrap();
                roi.set_max((
                    (full_size.0 as f32 * (click_pos.x - img_rsp.rect.min.x) / size.x) as u16,
                    (full_size.1 as f32 * (click_pos.y - img_rsp.rect.min.y) / size.y) as u16,
                ));
                roi.changed(true);
                retval = true;
            }
            else if img_rsp.secondary_clicked()
            {
                //right click
                let click_pos = img_rsp.interact_pointer_pos().unwrap();
                roi.set_min((
                    (full_size.0 as f32 * (click_pos.x - img_rsp.rect.min.x) / size.x) as u16,
                    (full_size.1 as f32 * (click_pos.y - img_rsp.rect.min.y) / size.y) as u16,
                ));
                roi.changed(true);
                retval = true;
            }
            else
            { //roi has not changed/ image not clicked
            }
            ui.painter_at(img_rsp.rect).rect_stroke(
                Rect {
                    min : Pos2::new(
                        roi_min.0 as f32 / full_size.0 as f32 * size.x + img_rsp.rect.min.x,
                        roi_min.1 as f32 / full_size.1 as f32 * size.y + img_rsp.rect.min.y,
                    ),
                    max : Pos2::new(
                        roi_max.0 as f32 / full_size.0 as f32 * size.x + img_rsp.rect.min.x,
                        roi_max.1 as f32 / full_size.1 as f32 * size.y + img_rsp.rect.min.y,
                    ),
                },
                0.0,
                Stroke::new(3.0, Color32::YELLOW),
            );
            ui.add_space(5.0);
            ui.vertical(|ui| {
                ui.add_space(6.0);
                ui.image(
                    self.legend.grad_legend().id(),
                    Vec2 {
                        x : 10.0,
                        y : size.y - 10.0,
                    },
                );
                ui.add_space(4.0);
            });
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = egui::vec2(0.0, 0.0);
                let ava_space = ui.available_height();
                let label_height = ui
                    .label(format!("{:.2}", self.scale[self.scale.len() - 1]))
                    .rect
                    .height();
                let n = ava_space / label_height;
                let (stride, label_count) = if n >= 33.0
                {
                    (1, 33)
                }
                else if n >= 17.0
                {
                    (2, 17)
                }
                else if n >= 9.0
                {
                    (4, 9)
                }
                else
                {
                    (8, 5)
                };
                let labels_height_sum = label_height * label_count as f32 + 2.0;
                let spacing_height = (ava_space - labels_height_sum) / (label_count - 1) as f32;
                for i in (0..(label_count - 1)).rev()
                {
                    ui.add_space(spacing_height);
                    ui.label(format!("{:.2}", self.scale[i * stride]));
                }
            });
        });
        (retval, responce)
    }
}
impl TTViewParams
{
    ///#Returns (a,b)
    /// a - any element changed?
    /// b - scale changed?
    fn show(&mut self, ui : &mut egui::Ui) -> (bool, bool)
    {
        let mut retval = (false, false);
        ui.horizontal_wrapped(|ui| {
            retval.0 |= self.show_combobox(ui);
            match self
            {
                WaveletView(params) =>
                {
                    ui.style_mut().wrap = Some(false);
                    ui.label("| mode:");
                    retval.0 |= params.display_mode.show_combobox(ui);
                    ui.style_mut().wrap = Some(false);
                    ui.label("| wavelet:");
                    retval.0 |= params.wavelet.show_combobox(ui);
                    ui.label("| scale:");
                    retval.1 |= params.scale.show(ui);
                    retval.0 |= retval.1;
                    ui.label("| frame:");
                    retval.0 |= params.time.show(ui);
                    retval.0 |= params.denoise.show_switchable(ui, "denoise");
                }
                TimeView(params) =>
                {
                    ui.label("| frame:");
                    retval.0 |= params.time.show(ui);
                    retval.0 |= params.denoise.show_switchable(ui, "denoise");
                }
                FourierView(params) =>
                {
                    ui.label("| mode:");
                    retval.0 |= params.display_mode.show_combobox(ui);
                    ui.label("| relative frequency:");
                    retval.0 |= params.freq.show(ui);
                    retval.0 |= params.denoise.show_switchable(ui, "denoise");
                }
            }
        });
        retval
    }
}
fn tt_view_new(
    name : &str,
    params : TTViewParams,
    ctx : &Context,
    roi : Arc<SemiAtomicRect>,
) -> (TTViewGUI, TTViewBackend)
{
    let (image_input, image_output) = triple_buffer(&Thermogram::new(ctx.load_texture(
        name,
        ColorImage::new([100, 100], Color32::TRANSPARENT),
        TextureOptions::LINEAR,
    )));
    let (params_input, params_output) = triple_buffer(&params);
    let state = Arc::new(AtomicTTViewState::new(TTViewState::Invalid));
    (
        TTViewGUI {
            state :  state.clone(),
            image :  image_output,
            params : params_input,
            roi :    roi.clone(),
        },
        TTViewBackend {
            state :      state,
            thermogram : image_input,
            params :     params_output,
            roi :        roi.clone(),
        },
    )
}

//=======================================
//============Implementations============
//=======================================

impl TTViewGUI
{
    pub fn show(&mut self, ui : &mut egui::Ui) -> bool
    {
        let mut retval = false;
        ui.vertical(|ui| {
            if self.state.load(Ordering::Relaxed) == TTViewState::Invalid
            {
                //grey out & block interactive elements of this view
                ui.set_enabled(false);
            }
            if let (true, scale_changed) = self.params.input_buffer().show(ui)
            {
                //params changed by user
                retval = true;
                if scale_changed
                {
                    if let WaveletView(mut params) = *self.params.input_buffer()
                    {
                        let out_count = params.scale.val - params.scale.min;
                        let out_count_half = out_count >> 1;
                        let out_count_half_larger = out_count - out_count_half;
                        params.time.max = params.scale.max - 1 - out_count_half_larger;
                        params.time.min = params.scale.min - 1 + out_count_half;
                        (*self.params.input_buffer()) = WaveletView(params);
                    }
                }

                self.state.store(TTViewState::Changed, Ordering::Relaxed);
                let temp = self.params.input_buffer().clone();
                self.params.publish();
                *self.params.input_buffer() = temp;
            }
            let gram = self.image.read();
            ui.with_layout(
                egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                |ui| {
                    match self.state.load(Ordering::Relaxed)
                    {
                        TTViewState::Valid =>
                        {
                            let (changed, _rsp) = gram.show(ui, &self.roi);
                            retval |= changed;
                        }
                        TTViewState::Processing | TTViewState::Changed =>
                        {
                            let (changed, rsp) = gram.show(ui, &self.roi);
                            retval |= changed;
                            ui.put(rsp.response.rect, Spinner::default());
                        }
                        TTViewState::Invalid => (),
                    }
                },
            );
        });
        retval
    }
}

impl TTStateGUI
{
    pub fn new(ctx : &egui::Context) -> Self
    {
        TTGradients::init_grad(ctx);
        let default_view_params = [
            TTViewParams::time_default(),
            TTViewParams::wavelet_wavelet(WaveletType::Morlet, ComplexResultMode::Phase),
            TTViewParams::wavelet_wavelet(WaveletType::Morlet, ComplexResultMode::Magnitude),
            TTViewParams::fourier_default(),
        ];
        let roi = Arc::new(SemiAtomicRect::new((0, 0), (1, 1), (1, 1)));
        // let (mut views_gui, mut views_backend) : ([TTViewGUI; 4], [TTViewBackend; 4]) =
        let [(g1,b1),(g2,b2),(g3,b3),(g4,b4)] //: [(TTViewGUI, TTViewBackend); 4] 
        = default_view_params.map( |x| tt_view_new("TTParams",x, ctx,roi.clone()));

        let (path_gui, path_backend) = triple_buffer(&None);

        let changed = Arc::new((Mutex::new(false), Condvar::new()));
        let stop_flag = Arc::new(AtomicBool::new(false));
        let frames = Arc::new(AtomicUsize::new(0));
        let state = Arc::new(AtomicFileState::new(FileState::None));
        TTStateGUI {
            views :          [g1, g2, g3, g4],
            changed :        changed.clone(),
            stop_flag :      stop_flag.clone(),
            file :           TTFileGUI {
                frames : frames.clone(),
                state :  state.clone(),
                path :   path_gui,
                roi :    roi.clone(),
            },
            backend_handle : Some(thread::spawn(move || {
                let backend_state = TTStateBackend::new(
                    [b1, b2, b3, b4],
                    changed,
                    stop_flag,
                    frames,
                    state,
                    path_backend,
                    roi.clone(),
                );
                backend_state.run();
            })),
        }
    }
    pub fn notify_backend(&self) -> ()
    {
        //wake up backend
        let &(ref lock, ref cvar) = &*self.changed;
        let mut started = lock.lock();
        *started = true;
        cvar.notify_one();
    }
    pub fn set_file_path(&mut self, path : Option<OsString>) -> ()
    {
        //invalidate views
        self.views
            .iter()
            .for_each(|view| view.state.store(TTViewState::Invalid, Ordering::Relaxed));
        //"send" updated path to backend
        self.file.path.write(path.clone());
        //update new gui working buffer
        *self.file.path.input_buffer() = path;
        self.file.state.store(FileState::New, Ordering::Relaxed);
        self.notify_backend()
    }
    pub fn get_file(&mut self) -> (Option<OsString>, FileState)
    {
        (
            (*self.file.path.input_buffer()).clone(),
            self.file.state.load(Ordering::Relaxed),
        )
    }
}
impl TTStateGUI
{
    pub fn show(&mut self, ui : &mut egui::Ui) -> ()
    {
        ui.vertical(|ui| {
            ui.heading("ThermoTransform v0.1");
            ui.horizontal(|ui| {
                ui.label("Input file: ");
                let (path, file_state) = self.get_file();
                match (path, file_state)
                {
                    (None, _) => (),
                    (_, FileState::None) => (),
                    (Some(path), FileState::Ready) =>
                    {
                        ui.label(path.to_string_lossy());
                    }
                    (Some(path), FileState::Loaded) =>
                    {
                        self.file
                            .state
                            .store(FileState::ProcessingWavelet, Ordering::Relaxed);
                        let frames = self.file.frames.load(Ordering::Relaxed);
                        //enable views generation
                        self.views.iter_mut().for_each(|view| {
                            let mut params = (*view.params.input_buffer()).clone();
                            match params
                            {
                                WaveletView(mut tparams) =>
                                {
                                    tparams.scale.max = frames;
                                    tparams.scale.val = 1;
                                    tparams.scale.min = 1;
                                    tparams.time.max = tparams.scale.max - tparams.scale.val;
                                    params = WaveletView(tparams)
                                }
                                TimeView(mut tparams) =>
                                {
                                    tparams.time.max = frames - 1;
                                    params = TimeView(tparams);
                                }
                                FourierView(mut tparams) =>
                                {
                                    tparams.freq.max = frames / 2;
                                    params = FourierView(tparams);
                                }
                            }
                            //update backend buffer
                            *view.params.input_buffer() = params;
                            view.params.publish();
                            //update gui working buffer
                            *view.params.input_buffer() = params;
                            view.state.store(TTViewState::Changed, Ordering::Relaxed);
                        });
                        self.notify_backend();
                        ui.label(path.to_string_lossy());
                        ui.label(" Processing...");
                        ui.spinner();
                    }
                    (Some(path), FileState::ProcessingWavelet) =>
                    {
                        ui.label(path.to_string_lossy());
                        ui.label(" Processing Wavelet transforms...");
                        ui.spinner();
                    }
                    (Some(path), FileState::ProcessingFourier) =>
                    {
                        ui.label(path.to_string_lossy());
                        ui.label(" Processing Fourier transforms...");
                        ui.spinner();
                    }
                    (Some(path), _) =>
                    {
                        ui.label(path.to_string_lossy());
                        ui.label(" Loading...");
                        ui.spinner();
                    }
                }
                if ui.button("…").clicked()
                {
                    let future = async {
                        let file = rfd::AsyncFileDialog::new().pick_file().await;

                        let data = file;
                        if let Some(path) = data
                        {
                            #[cfg(not(target_arch = "wasm32"))]
                            {
                                self.set_file_path(Some(path.inner().into()));
                            }
                            #[cfg(target_arch = "wasm32")]
                            {
                                self.set_file_path(path.inner());
                            }
                        }
                    };
                    futures::executor::block_on(future);
                }
            });
            let available_height = ui.available_height() / 2.0;
            let available_width = ui.available_width() / 2.0;
            let mut changed = false;
            TableBuilder::new(ui)
                .column(Column::exact(available_width))
                .column(Column::exact(available_width))
                .body(|mut body| {
                    body.row(available_height, |mut row| {
                        row.col(|ui| {
                            changed |= self.views[0].show(ui);
                        });
                        row.col(|ui| {
                            changed |= self.views[1].show(ui);
                        });
                    });
                    body.row(available_height, |mut row| {
                        row.col(|ui| {
                            changed |= self.views[2].show(ui);
                        });
                        row.col(|ui| {
                            changed |= self.views[3].show(ui);
                        });
                    });
                });
            if changed
            {
                if self.file.roi.changed(false)
                {
                    //if roi has changed refresh all views
                    self.views.iter().for_each(|view| {
                        view.state.store(TTViewState::Changed, Ordering::Relaxed);
                    })
                }
                self.notify_backend();
            }
        });
    }
}
impl Drop for TTStateGUI
{
    fn drop(&mut self)
    {
        self.stop_flag.store(true, Ordering::SeqCst);
        self.notify_backend();
        loop
        {
            if let Some(handle) = &self.backend_handle
            {
                if handle.is_finished()
                {
                    break;
                }
                else
                {
                    continue;
                }
            }
            else
            {
                break;
            }
        }
    }
}
