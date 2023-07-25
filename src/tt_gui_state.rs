use crate::tt_backend_state::*;
use crate::tt_common::*;
use crate::tt_file::TTFile;
use crate::wavelet::WaveletType;
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
use ndarray::IntoDimension;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use strum::VariantNames;
use triple_buffer as tribuf;
use triple_buffer::triple_buffer;
use Ordering::Relaxed;

//=======================================
//=================Types=================
//=======================================

pub struct TTViewGUI
{
    state :     Arc<AtomicTTViewState>,
    image :     tribuf::Output<Thermogram>,
    view_mode : Arc<ViewMode>,
    settings :  Arc<GlobalSettings>,
}

pub struct TTFileGUI
{
    state : Arc<AtomicFileState>,
    path :  tribuf::Input<Option<TTFile>>,
}

pub struct TTStateGUI
{
    views :          [TTViewGUI; 4],
    changed :        Arc<(Mutex<bool>, Condvar)>,
    stop_flag :      Arc<AtomicBool>,
    file :           TTFileGUI,
    backend_handle : Option<JoinHandle<()>>,
    settings :       Arc<GlobalSettings>,
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
pub trait AtomicBoolSwitchable<T>
{
    fn show_switchable(&self, ui : &mut egui::Ui, text : &str) -> bool;
}
impl AtomicBoolSwitchable<AtomicBool> for AtomicBool
{
    fn show_switchable(&self, ui : &mut egui::Ui, text : &str) -> bool
    {
        let mut retval = false;
        let self_val = self.load(Ordering::Relaxed);
        let text = if self_val == true
        {
            RichText::new(text)
        }
        else
        {
            RichText::new(text).strikethrough()
        };
        if ui.button(text).clicked()
        {
            self.store(!self_val, Ordering::Relaxed);
            retval = true;
        }
        retval
    }
}
pub trait EnumCombobox<T>
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

macro_rules! atomicCombobox {
    ($atomic:expr,$ui:ident) => {{
        let mut _atomicCombobox = $atomic.load(Ordering::Relaxed);
        let _ret = _atomicCombobox.show_combobox($ui);
        $atomic.store(_atomicCombobox, Ordering::Relaxed);
        _ret
    }};
}
impl ViewMode
{
    pub fn controls(&self, global : &GlobalSettings, ui : &mut egui::Ui) -> bool
    {
        let mut changed = false;

        ui.horizontal_wrapped(|ui| {
            if atomicCombobox!(self.domain, ui)
            {
                //domain has changed: clip mode_counter
                let mode_axes = Self::ViewModeAxes[self.domain.load(Ordering::Relaxed) as usize];
                let mut mode_counter = self.mode_counter.load(Ordering::Relaxed);
                mode_counter = mode_counter % mode_axes.len();
                self.mode_counter.store(mode_counter, Ordering::Relaxed);
                changed = true;
            }
            let mode_axes = Self::ViewModeAxes[self.domain.load(Ordering::Relaxed) as usize];
            let mut mode_counter = self.mode_counter.load(Ordering::Relaxed);
            let settings_axes = mode_axes[mode_counter].1;
            /*view mode button*/
            {
                let [a, b] = self.get_view_axes();
                let text = format!("{}-{}", Into::<char>::into(a), Into::<char>::into(b));
                if ui.button(text).clicked()
                {
                    mode_counter = (mode_counter + 1) % mode_axes.len();
                    self.mode_counter.store(mode_counter, Ordering::Relaxed);
                    changed = true;
                }
            }
            changed |= self.bind_position.show_switchable(ui, "🔗");
            match self.domain.load(Ordering::Relaxed)
            {
                ViewModeDomain::TimeView =>
                {}
                ViewModeDomain::FourierView =>
                {
                    ui.style_mut().wrap = Some(false);
                    ui.label("| mode:");
                    changed |= atomicCombobox!(self.display_mode, ui);
                }
                ViewModeDomain::FastWaveletView | ViewModeDomain::WaveletView =>
                {
                    ui.style_mut().wrap = Some(false);
                    ui.label("| mode:");
                    changed |= atomicCombobox!(self.display_mode, ui);
                    ui.style_mut().wrap = Some(false);
                    ui.label("| wavelet:");
                    changed |= atomicCombobox!(self.wavelet, ui);
                }
            }
            /*position DragValues*/
            {
                let mut position = self.position.write();
                if self.bind_position.load(Ordering::Relaxed)
                {
                    let gp = *global.crossection.read();
                    settings_axes.iter().for_each(|&i| {
                        changed |= gp[i as usize] != position[i as usize];
                    });
                    *position = gp;
                }
                let full_size = global.full_size.read();
                for &axis in settings_axes
                {
                    let uaxis = axis as usize;
                    let mut ranged_val = if axis == TTAxis::S
                    {
                        RangedVal {
                            val : position[uaxis] + 1,
                            min : 1,
                            max : full_size[uaxis] + 1,
                        }
                    }
                    else
                    {
                        RangedVal {
                            val : position[uaxis],
                            min : 0,
                            max : full_size[uaxis],
                        }
                    };
                    ui.label(Into::<&str>::into(axis));
                    changed |= ranged_val.show(ui);
                    position[uaxis] = ranged_val.val;
                    if axis == TTAxis::S
                    {
                        position[uaxis] -= 1;
                    }
                }
                if self.bind_position.load(Ordering::Relaxed)
                {
                    *global.crossection.write() = *position;
                }
            }
            changed |= self.denoise.show_switchable(ui, "denoise");
        });
        changed
    }
}
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
        settings : &GlobalSettings,
        view_axes : [TTAxis; 2],
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
            let mut roi_x = settings.get_roi(view_axes[0]);
            let mut roi_y = settings.get_roi(view_axes[1]);
            let full_size_x = settings.full_size.read()[view_axes[0] as usize];
            let full_size_y = settings.full_size.read()[view_axes[1] as usize];
            let roi_zoom = settings.roi_zoom.load(Ordering::Relaxed);
            if img_rsp.clicked()
            {
                //left click- convert click_position to data_position
                let select_mode = settings.select_mode.load(Relaxed);
                let click_pos = img_rsp.interact_pointer_pos().unwrap();
                let (mut new_x, mut new_y) = if roi_zoom
                {
                    let (width, height) = (roi_x.len(), roi_y.len());

                    (
                        (roi_x.start as f32
                            + width as f32 * (click_pos.x - img_rsp.rect.min.x) / size.x)
                            as usize,
                        (roi_y.start as f32
                            + height as f32 * (click_pos.y - img_rsp.rect.min.y) / size.y)
                            as usize,
                    )
                }
                else
                {
                    //roi zoom not enabled
                    (
                        (full_size_x as f32 * (click_pos.x - img_rsp.rect.min.x) / size.x) as usize,
                        (full_size_y as f32 * (click_pos.y - img_rsp.rect.min.y) / size.y) as usize,
                    )
                };
                new_x = new_x.clamp(0, full_size_x - 1);
                new_y = new_y.clamp(0, full_size_y - 1);
                if select_mode == SelectMode::Crossection
                {
                    let mut crossection = settings.crossection.write();
                    crossection[view_axes[0] as usize] = new_x;
                    crossection[view_axes[1] as usize] = new_y;
                }
                else
                {
                    //selecting ROI
                    let (other_x, other_y) = if select_mode == SelectMode::RoiMax
                    {
                        (roi_x.start.clone(), roi_y.start.clone())
                    }
                    else
                    {
                        //SelectMode::RoiMin
                        (roi_x.end.clone(), roi_y.end.clone())
                    };
                    roi_x.start = usize::min(new_x, other_x);
                    roi_y.start = usize::min(new_y, other_y);
                    roi_x.end = usize::max(new_x, other_x);
                    roi_y.end = usize::max(new_y, other_y);
                    let mut roi_min = settings.roi_min.write();
                    let mut roi_max = settings.roi_max.write();
                    roi_min[view_axes[0] as usize] = roi_x.start;
                    roi_min[view_axes[1] as usize] = roi_y.start;
                    roi_max[view_axes[0] as usize] = roi_x.end;
                    roi_max[view_axes[1] as usize] = roi_y.end;
                }
                settings.changed(true);
                retval = true;
            }

            let crossection_x = settings.crossection.read()[view_axes[0] as usize];
            let crossection_y = settings.crossection.read()[view_axes[1] as usize];
            if roi_zoom
            {
                let (width, height) = (roi_x.len(), roi_y.len());
                ui.painter_at(img_rsp.rect).rect_stroke(
                    img_rsp.rect,
                    0.0,
                    Stroke::new(3.0, Color32::YELLOW),
                );
                ui.painter_at(img_rsp.rect).hline(
                    0.0..=size.x + img_rsp.rect.min.x,
                    (crossection_y as f32 - roi_y.start as f32) / height as f32 * size.y
                        + img_rsp.rect.min.y,
                    Stroke::new(3.0, Color32::GREEN),
                );
                ui.painter_at(img_rsp.rect).vline(
                    (crossection_x as f32 - roi_x.start as f32) / width as f32 * size.x
                        + img_rsp.rect.min.x,
                    0.0..=size.y + img_rsp.rect.min.y,
                    Stroke::new(3.0, Color32::GREEN),
                );
            }
            else
            {
                ui.painter_at(img_rsp.rect).rect_stroke(
                    Rect {
                        min : Pos2::new(
                            roi_x.start as f32 / full_size_x as f32 * size.x + img_rsp.rect.min.x,
                            roi_y.start as f32 / full_size_y as f32 * size.y + img_rsp.rect.min.y,
                        ),
                        max : Pos2::new(
                            roi_x.end as f32 / full_size_x as f32 * size.x + img_rsp.rect.min.x,
                            roi_y.end as f32 / full_size_y as f32 * size.y + img_rsp.rect.min.y,
                        ),
                    },
                    0.0,
                    Stroke::new(3.0, Color32::YELLOW),
                );
                ui.painter_at(img_rsp.rect).hline(
                    0.0..=size.x + img_rsp.rect.min.x,
                    crossection_y as f32 / full_size_y as f32 * size.y + img_rsp.rect.min.y,
                    Stroke::new(3.0, Color32::GREEN),
                );
                ui.painter_at(img_rsp.rect).vline(
                    crossection_x as f32 / full_size_x as f32 * size.x + img_rsp.rect.min.x,
                    0.0..=size.y + img_rsp.rect.min.y,
                    Stroke::new(3.0, Color32::GREEN),
                );
            };
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

fn tt_view_new(
    name : &str,
    params : ViewMode,
    ctx : &Context,
    settings : Arc<GlobalSettings>,
) -> (TTViewGUI, TTViewBackend)
{
    let (image_input, image_output) = triple_buffer(&Thermogram::new(ctx.load_texture(
        name,
        ColorImage::new([100, 100], Color32::TRANSPARENT),
        TextureOptions::LINEAR,
    )));
    let aparams = Arc::new(params.clone());
    let state = Arc::new(AtomicTTViewState::new(TTViewState::Invalid));
    (
        TTViewGUI {
            state :     state.clone(),
            image :     image_output,
            view_mode : aparams.clone(),
            settings :  settings.clone(),
        },
        TTViewBackend {
            state :            state,
            thermogram :       image_input,
            view_mode :        aparams,
            settings :         settings.clone(),
            frozen_view_mode : params,
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
            if self.view_mode.controls(&self.settings, ui)
            {
                //params changed by user
                retval = true;
                self.state.store(TTViewState::Changed, Ordering::Relaxed);
            }
            let gram = self.image.read();
            ui.with_layout(
                egui::Layout::centered_and_justified(egui::Direction::LeftToRight),
                |ui| {
                    match self.state.load(Ordering::Relaxed)
                    {
                        TTViewState::Valid =>
                        {
                            let (changed, _rsp) =
                                gram.show(ui, &self.settings, self.view_mode.get_view_axes());
                            retval |= changed;
                        }
                        TTViewState::Processing | TTViewState::Changed =>
                        {
                            let (changed, rsp) =
                                gram.show(ui, &self.settings, self.view_mode.get_view_axes());
                            retval |= changed;
                            ui.put(rsp.response.rect, Spinner::default());
                            ui.ctx().request_repaint(); //faster next screen refresh, when waiting for new view
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
    pub fn new(ctx : &egui::Context, file : Option<TTFile>) -> Self
    {
        TTGradients::init_grad(ctx);
        let default_view_params = [
            ViewMode::new(
                ViewModeDomain::TimeView,
                Default::default(),
                Default::default(),
            ),
            ViewMode::new(
                ViewModeDomain::FastWaveletView,
                WaveletType::Morlet,
                ComplexResultMode::Phase,
            ),
            ViewMode::new(
                ViewModeDomain::FastWaveletView,
                WaveletType::Morlet,
                ComplexResultMode::Magnitude,
            ),
            ViewMode::new(
                ViewModeDomain::FourierView,
                Default::default(),
                ComplexResultMode::Phase,
            ),
        ];
        let settings = Arc::new(GlobalSettings::default());
        // let (mut views_gui, mut views_backend) : ([TTViewGUI; 4], [TTViewBackend; 4]) =
        let [(g1,b1),(g2,b2),(g3,b3),(g4,b4)] //: [(TTViewGUI, TTViewBackend); 4] 
        = default_view_params.map( |x| tt_view_new("TTParams",x, ctx,settings.clone()));

        let (path_gui, path_backend);
        let state;
        if let Some(file) = file
        {
            (path_gui, path_backend) = triple_buffer(&Some(file));
            state = Arc::new(AtomicFileState::new(FileState::New));
        }
        else
        {
            (path_gui, path_backend) = triple_buffer(&None);
            state = Arc::new(AtomicFileState::new(FileState::None));
        }

        let changed = Arc::new((Mutex::new(false), Condvar::new()));
        let stop_flag = Arc::new(AtomicBool::new(false));
        TTStateGUI {
            views :          [g1, g2, g3, g4],
            changed :        changed.clone(),
            stop_flag :      stop_flag.clone(),
            file :           TTFileGUI {
                state : state.clone(),
                path :  path_gui,
            },
            settings :       settings.clone(),
            backend_handle : Some(thread::spawn(move || {
                let backend_state = TTStateBackend::new(
                    [b1, b2, b3, b4],
                    changed,
                    stop_flag,
                    state,
                    path_backend,
                    settings.clone(),
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
    pub fn set_file_path(&mut self, path : Option<TTFile>) -> ()
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
    pub fn get_file(&mut self) -> (Option<TTFile>, FileState)
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
        let mut changed = false;
        ui.vertical(|ui| {
            let header = format!("ThermoTransform {}", env!("CARGO_PKG_VERSION"));
            ui.heading(header);
            ui.horizontal(|ui| {
                ui.label("Input file: ");
                let (path, file_state) = self.get_file();
                match (path, file_state)
                {
                    (None, _) => (),
                    (_, FileState::None) => (),
                    (Some(path), FileState::ReadySaving) =>
                    {
                        ui.label(path.path());
                        ui.label(" Saving...");
                        ui.spinner();
                    }
                    (Some(path), FileState::Ready) =>
                    {
                        ui.label(path.path());
                    }
                    (Some(path), FileState::Loaded) =>
                    {
                        self.file
                            .state
                            .store(FileState::ProcessingFourier, Ordering::Relaxed);
                        //enable views generation
                        self.views.iter_mut().for_each(|view| {
                            view.state.store(TTViewState::Changed, Ordering::Relaxed);
                            *view.view_mode.position.write() = [0, 0, 0, 0, 0].into_dimension();
                        });
                        self.notify_backend();
                        ui.label(path.path());
                        ui.label(" Processing...");
                        ui.spinner();
                    }
                    (Some(path), FileState::ProcessingWavelet) =>
                    {
                        ui.label(path.path());
                        ui.label(" Processing Wavelet transforms...");
                        ui.spinner();
                    }
                    (Some(path), FileState::ProcessingFourier) =>
                    {
                        ui.label(path.path());
                        ui.label(" Processing Fourier transforms...");
                        ui.spinner();
                    }
                    (Some(path), FileState::Error) =>
                    {
                        ui.label(path.path());
                        ui.label(
                            RichText::new(" !!! Invalid file !!!")
                                .color(Color32::RED)
                                .strong(),
                        );
                        ui.spinner();
                    }
                    (Some(path), _) =>
                    {
                        ui.label(path.path());
                        ui.label(" Loading...");
                        ui.spinner();
                    }
                }
                if ui.button("…").clicked()
                {
                    if let Some(path) = TTFile::new_from_file_dialog()
                    {
                        self.set_file_path(Some(path));
                    }
                }
                if self.settings.roi_zoom.show_switchable(ui, "zoom ROI")
                {
                    changed = true;
                    self.settings.changed(true);
                }
                changed |= atomicCombobox!(self.settings.select_mode, ui);
            });
            let available_height = ui.available_height() / 2.0;
            let available_width = ui.available_width() / 2.0;
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
                ui.ctx().request_repaint(); //speed up next screen refresh
                if self.settings.changed(false)
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
