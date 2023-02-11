use egui::{ColorImage, Context, DragValue, TextureHandle, TextureOptions, Vec2};
use egui_extras::{Column, TableBuilder};
use parking_lot::{Condvar, Mutex};
use std::ffi::OsString;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use strum::VariantNames;
use triple_buffer as tribuf;
use triple_buffer::triple_buffer;

use crate::cwt::*;
use crate::tt_backend_state::*;
use crate::tt_common_state::*;
use crate::tt_input_data::*;

//=======================================
//=================Types=================
//=======================================

pub struct TTViewGUI
{
    state :  Arc<AtomicTTViewState>,
    image :  tribuf::Output<TextureHandle>,
    params : tribuf::Input<TTViewParams>,
}

pub struct TTFileGUI
{
    frames : Arc<AtomicUsize>,
    state :  Arc<AtomicFileState>,
    path :   tribuf::Input<Option<OsString>>,
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
                TransformView {
                    scale,
                    time,
                    wavelet,
                    mode,
                } =>
                {
                    ui.style_mut().wrap = Some(false);
                    ui.label("| mode:");
                    retval.0 |= mode.show_combobox(ui);
                    ui.style_mut().wrap = Some(false);
                    ui.label("| wavelet:");
                    retval.0 |= wavelet.show_combobox(ui);
                    ui.label("| scale:");
                    retval.1 |= scale.show(ui);
                    retval.0 |= retval.1;
                    ui.label("| frame:");
                    retval.0 |= time.show(ui);
                }
                TimeView { time } =>
                {
                    ui.label("| frame:");
                    retval.0 |= time.show(ui);
                }
            }
        });
        retval
    }
}
fn tt_view_new(name : &str, params : TTViewParams, ctx : &Context) -> (TTViewGUI, TTViewBackend)
{
    let (image_input, image_output) =
        triple_buffer(&ctx.load_texture(name, ColorImage::example(), TextureOptions::LINEAR));
    let (params_input, params_output) = triple_buffer(&params);
    let state = Arc::new(AtomicTTViewState::new(TTViewState::Invalid));
    (
        TTViewGUI {
            state :  state.clone(),
            image :  image_output,
            params : params_input,
        },
        TTViewBackend {
            state :  state,
            image :  image_input,
            params : params_output,
        },
    )
}

//=======================================
//============Implementations============
//=======================================

impl TTViewGUI
{
    pub fn show(&mut self, ui : &mut egui::Ui) -> ()
    {
        ui.vertical(|ui| {
            if self.state.load(Ordering::Relaxed) == TTViewState::Invalid
            {
                //grey out & block interactive elements of this view
                ui.set_enabled(false);
            }
            if let (true, scale_changed) = self.params.input_buffer().show(ui)
            {
                //params changed by user
                if scale_changed
                {
                    if let TransformView {
                        scale,
                        mut time,
                        wavelet,
                        mode,
                    } = *self.params.input_buffer()
                    {
                        time.max = scale.max - scale.val;
                        (*self.params.input_buffer()) = TransformView {
                            scale,
                            time,
                            wavelet,
                            mode,
                        };
                    }
                }

                self.state.store(TTViewState::Changed, Ordering::Relaxed);
                let temp = self.params.input_buffer().clone();
                self.params.publish();
                *self.params.input_buffer() = temp;
            }
            let image = self.image.read();
            ui.with_layout(
                egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                |ui| {
                    let available_size = ui.available_size();
                    let image_aspect = image.aspect_ratio();
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
                    match self.state.load(Ordering::Relaxed)
                    {
                        TTViewState::Valid =>
                        {
                            ui.image(image.id(), size);
                        }
                        TTViewState::Processing | TTViewState::Changed =>
                        {
                            ui.spinner();
                        }
                        TTViewState::Invalid => (),
                    }
                },
            );
        });
    }
}

impl TTStateGUI
{
    pub fn new(ctx : &egui::Context) -> Self
    {
        let default_view_params = [
            TimeView {
                time : RangedVal::default(),
            },
            TransformView {
                scale :   RangedVal::default(),
                time :    RangedVal::default(),
                wavelet : WaveletType::Morlet,
                mode :    WtResultMode::Phase,
            },
            TransformView {
                scale :   RangedVal::default(),
                time :    RangedVal::default(),
                wavelet : WaveletType::Morlet,
                mode :    WtResultMode::Magnitude,
            },
            TimeView {
                time : RangedVal::default(),
            },
        ];
        // let (mut views_gui, mut views_backend) : ([TTViewGUI; 4], [TTViewBackend; 4]) =
        let [(g1,b1),(g2,b2),(g3,b3),(g4,b4)] //: [(TTViewGUI, TTViewBackend); 4] 
        = default_view_params.map( |x| tt_view_new("TTParams",x, ctx));

        let (path_gui, path_backend) = triple_buffer(&None);

        let backend = TTStateBackend {
            views :     [b1, b2, b3, b4],
            changed :   Arc::new((Mutex::new(false), Condvar::new())),
            stop_flag : Arc::new(AtomicBool::new(false)),
            file :      TTFileBackend {
                frames :           Arc::new(AtomicUsize::new(0)),
                state :            Arc::new(AtomicFileState::new(FileState::None)),
                path :             path_backend,
                input_data :       None,
                input_integrated : None,
            },
        };
        TTStateGUI {
            views :          [g1, g2, g3, g4],
            changed :        backend.changed.clone(),
            stop_flag :      backend.stop_flag.clone(),
            file :           TTFileGUI {
                frames : backend.file.frames.clone(),
                state :  backend.file.state.clone(),
                path :   path_gui,
            },
            backend_handle : Some(thread::spawn(move || {
                let mut backend_state = backend;
                backend_state.run();
            })),
        }
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
        //wake up backend
        {
            let &(ref lock, ref cvar) = &*self.changed;
            let mut started = lock.lock();
            *started = true;
            cvar.notify_one();
        }
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
                            .store(FileState::Processing, Ordering::Relaxed);
                        let frames = self.file.frames.load(Ordering::Relaxed);
                        //enable views generation
                        self.views.iter_mut().for_each(|view| {
                            let mut params = (*view.params.input_buffer()).clone();
                            match params
                            {
                                TransformView {
                                    mut scale,
                                    mut time,
                                    wavelet,
                                    mode,
                                } =>
                                {
                                    scale.max = frames;
                                    scale.val = 1;
                                    scale.min = 1;
                                    time.max = scale.max - scale.val;
                                    params = TransformView {
                                        scale,
                                        time,
                                        wavelet,
                                        mode,
                                    }
                                }
                                TimeView { mut time } =>
                                {
                                    time.max = frames - 1;
                                    params = TimeView { time };
                                }
                            }
                            //update backend buffer
                            *view.params.input_buffer() = params;
                            view.params.publish();
                            //update gui working buffer
                            *view.params.input_buffer() = params;
                            view.state.store(TTViewState::Changed, Ordering::Relaxed);
                        });
                        ui.label(path.to_string_lossy());
                        ui.label(" Processing...");
                        ui.spinner();
                    }
                    (Some(path), FileState::Processing) =>
                    {
                        ui.label(path.to_string_lossy());
                        ui.label(" Processing...");
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
            TableBuilder::new(ui)
                .column(Column::exact(available_width))
                .column(Column::exact(available_width))
                .body(|mut body| {
                    body.row(available_height, |mut row| {
                        row.col(|ui| {
                            self.views[0].show(ui);
                        });
                        row.col(|ui| {
                            self.views[1].show(ui);
                        });
                    });
                    body.row(available_height, |mut row| {
                        row.col(|ui| {
                            self.views[2].show(ui);
                        });
                        row.col(|ui| {
                            self.views[3].show(ui);
                        });
                    });
                });
        });
    }
}
impl Drop for TTStateGUI
{
    fn drop(&mut self)
    {
        self.stop_flag.store(true, Ordering::SeqCst);
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
