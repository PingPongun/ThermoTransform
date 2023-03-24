#![warn(clippy::all, rust_2018_idioms)]
#![allow(nonstandard_style)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
use eframe::IconData;
use image;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main()
{
    // Log to stdout (if you run with `RUST_LOG=debug`).
    tracing_subscriber::fmt::init();
    let icon_raw = include_bytes!("../icon.ico");
    let icon = image::load_from_memory_with_format(icon_raw.as_slice(), image::ImageFormat::Ico)
        .unwrap()
        .to_rgba8();
    let (icon_width, icon_height) = icon.dimensions();
    let native_options = eframe::NativeOptions {
        drag_and_drop_support : true,
        icon_data : Some(IconData {
            rgba :   icon.into_raw(),
            width :  icon_width,
            height : icon_height,
        }),
        ..Default::default()
    };
    let _ = eframe::run_native(
        "ThermoTransform",
        native_options,
        Box::new(|cc| Box::new(ThermoTransform::ThermoTransformApp::new(cc))),
    );
}

// when compiling to web using trunk.
#[cfg(target_arch = "wasm32")]
fn main()
{
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    let web_options = eframe::WebOptions::default();
    eframe::start_web(
        "the_canvas_id", // hardcode it
        web_options,
        Box::new(|cc| Box::new(ThermoTransform::TemplateApp::new(cc))),
    )
    .expect("failed to start eframe");
}
