use std::path::PathBuf;
#[path = "thermo_backend.rs"]
mod thermo_backend;
/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct TemplateApp
{
    // Example stuff:
    label :         String,
    dropped_files : Option<PathBuf>,
    picked_path :   Option<String>,

    // this how you opt-out of serialization of a member
    #[serde(skip)]
    value :   f32,
    #[serde(skip)]
    backend : thermo_backend::TTState,
}

impl Default for TemplateApp
{
    fn default() -> Self
    {
        Self {
            // Example stuff:
            label :         "Hello World!".to_owned(),
            value :         2.7,
            dropped_files : None,
            picked_path :   None,
            backend :       thermo_backend::TTState::new(),
        }
    }
}

impl TemplateApp
{
    /// Called once before the first frame.
    pub fn new(cc : &eframe::CreationContext<'_>) -> Self
    {
        // This is also where you can customized the look at feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage
        {
            return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        }

        Default::default()
    }
}

impl eframe::App for TemplateApp
{
    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage : &mut dyn eframe::Storage)
    {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx : &egui::Context, _frame : &mut eframe::Frame)
    {
        let Self {
            label,
            value,
            dropped_files: _,
            picked_path: _,
            backend,
        } = self;

        // Examples of how to create different panels and windows.
        // Pick whichever suits you.
        // Tip: a good default choice is to just keep the `CentralPanel`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        #[cfg(not(target_arch = "wasm32"))] // no File->Quit on web pages!
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Quit").clicked()
                    {
                        _frame.close();
                    }
                });
            });
        });

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Side Panel");

            ui.horizontal(|ui| {
                ui.label("Write something: ");
                ui.text_edit_singleline(label);
            });

            ui.add(egui::Slider::new(value, 0.0..=10.0).text("value"));
            if ui.button("Increment").clicked()
            {
                *value += 1.0;
            }

            if ui.button("Open file…").clicked()
            {
                let future = async {
                    let file = rfd::AsyncFileDialog::new().pick_file().await;

                    let data = file;
                    if let Some(path) = data
                    {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            self.picked_path = Some(path.inner().display().to_string());
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            self.picked_path = Some(path.inner().as_string().unwrap());
                        }
                    }
                };
                futures::executor::block_on(future);
            }

            if let Some(picked_path) = &self.picked_path
            {
                ui.horizontal(|ui| {
                    ui.label("Picked file:");
                    ui.monospace(picked_path);
                });
            }

            #[cfg(not(target_arch = "wasm32"))]
            {
                // Show dropped files (if any):
                // {
                //     ui.group(|ui|
                //     {
                //         ui.label("Dropped files:");
                //         match &self.dropped_files {
                //             Some(val) => {ui.label(val.to_str().unwrap());},
                //             None => {ui.label("Not selected");},
                //         }
                //     });
                // }

                preview_files_being_dropped(ctx);

                // Collect dropped files:
                if !ctx.input().raw.dropped_files.is_empty()
                {
                    self.picked_path = Some(
                        ctx.input()
                            .raw
                            .dropped_files
                            .last()
                            .unwrap()
                            .clone()
                            .path
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .to_string(),
                    );
                }
            }
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label("powered by ");
                    ui.hyperlink_to("egui", "https://github.com/emilk/egui");
                    ui.label(" and ");
                    ui.hyperlink_to(
                        "eframe",
                        "https://github.com/emilk/egui/tree/master/crates/eframe",
                    );
                    ui.label(".");
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's

            ui.heading("eframe template");
            ui.hyperlink("https://github.com/emilk/eframe_template");
            ui.add(egui::github_link_file!(
                "https://github.com/emilk/eframe_template/blob/master/",
                "Source code."
            ));
            egui::warn_if_debug_build(ui);
        });

        if false
        {
            egui::Window::new("Window").show(ctx, |ui| {
                ui.label("Windows can be moved by dragging them.");
                ui.label("They are automatically sized based on contents.");
                ui.label("You can turn on resizing and scrolling if you like.");
                ui.label("You would normally chose either panels OR windows.");
            });
        }
    }
}

/// Preview hovering files:
#[cfg(not(target_arch = "wasm32"))]
fn preview_files_being_dropped(ctx : &egui::Context)
{
    use egui::*;
    use std::fmt::Write as _;

    if !ctx.input().raw.hovered_files.is_empty()
    {
        let mut text = "Dropping files:\n".to_owned();
        for file in &ctx.input().raw.hovered_files
        {
            if let Some(path) = &file.path
            {
                write!(text, "\n{}", path.display()).ok();
            }
            else if !file.mime.is_empty()
            {
                write!(text, "\n{}", file.mime).ok();
            }
            else
            {
                text += "\n???";
            }
        }

        let painter =
            ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("file_drop_target")));

        let screen_rect = ctx.input().screen_rect();
        painter.rect_filled(screen_rect, 0.0, Color32::from_black_alpha(192));
        painter.text(
            screen_rect.center(),
            Align2::CENTER_CENTER,
            text,
            TextStyle::Heading.resolve(&ctx.style()),
            Color32::WHITE,
        );
    }
}
