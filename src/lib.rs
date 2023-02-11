#![warn(clippy::all, rust_2018_idioms)]

mod app;
pub use app::ThermoTransformApp;
mod cwt;
mod thermo_backend;
mod tt_input_data;
