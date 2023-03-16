#![warn(clippy::all, rust_2018_idioms)]
#![allow(nonstandard_style)]
mod app;
pub use app::ThermoTransformApp;
mod cwt;
mod tt_backend_state;
mod tt_common;
mod tt_fourier;
mod tt_gui_state;
mod tt_input_data;
mod wavelet;
