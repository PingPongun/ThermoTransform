use crate::tt_gui_state::TTStateGUI;

pub struct ThermoTransformApp
{
    pub backend : TTStateGUI,
}

impl ThermoTransformApp
{
    /// Called once before the first frame.
    pub fn new(cc : &eframe::CreationContext<'_>) -> Self
    {
        // This is also where you can customized the look at feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        // if let Some(storage) = cc.storage
        // {
        //     return eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        // }
        Self {
            backend : TTStateGUI::new(&cc.egui_ctx),
        }
    }
}

impl eframe::App for ThermoTransformApp
{
    /// Called by the frame work to save state before shutdown.
    // fn save(&mut self, storage : &mut dyn eframe::Storage)
    // {
    //     eframe::set_value(storage, eframe::APP_KEY, self);
    // }

    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx : &egui::Context, _frame : &mut eframe::Frame)
    {
        let Self { backend } = self;

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

        egui::CentralPanel::default().show(ctx, |ui| {
            // The central panel the region left after adding TopPanel's and SidePanel's

            #[cfg(not(target_arch = "wasm32"))]
            {
                preview_files_being_dropped(ctx);

                // Collect dropped files:
                if !ctx.input(|i| i.raw.dropped_files.is_empty())
                {
                    backend.set_file_path(Some(ctx.input(|i| {
                        i.raw
                            .dropped_files
                            .last()
                            .unwrap()
                            .clone()
                            .path
                            .unwrap()
                            .into()
                    })));
                }
            }
            backend.show(ui);
        });
    }
}

/// Preview hovering files:
#[cfg(not(target_arch = "wasm32"))]
fn preview_files_being_dropped(ctx : &egui::Context)
{
    use egui::*;
    use std::fmt::Write as _;

    if !ctx.input(|i| i.raw.hovered_files.is_empty())
    {
        let text = ctx.input(|i| {
            let mut text = "Dropping files:\n".to_owned();
            for file in &i.raw.hovered_files
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
            text
        });

        let painter =
            ctx.layer_painter(LayerId::new(Order::Foreground, Id::new("file_drop_target")));

        let screen_rect = ctx.input(|i| i.screen_rect());
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
