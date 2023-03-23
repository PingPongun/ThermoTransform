use atomic_enum::atomic_enum;
use egui::{ColorImage, Context, TextureHandle, TextureOptions};
use lazy_static::*;

#[cfg(feature = "time_meas")]
use std::fs::File;
#[cfg(feature = "time_meas")]
use std::io::BufWriter;
#[cfg(feature = "time_meas")]
use std::io::Write;
use std::iter;
use std::mem::ManuallyDrop;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use strum::VariantNames;
use strum_macros::{EnumString, EnumVariantNames};

use crate::wavelet::WaveletType;

//=======================================
//=================Types=================
//=======================================

#[atomic_enum]
#[derive(PartialEq)]
pub enum FileState
{
    None,
    New,
    Loading,
    Loaded,
    ProcessingFourier,
    ProcessingWavelet,
    Ready,
}
#[derive(
    Clone, Copy, PartialEq, Debug, Default, strum_macros::AsRefStr, EnumString, EnumVariantNames,
)]
#[strum(serialize_all = "title_case")]
pub enum ComplexResultMode
{
    #[default]
    Phase,
    Magnitude,
    Real,
    Imaginary,
}
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct RangedVal
{
    pub val : usize,
    pub min : usize,
    pub max : usize,
}
///(x,y)
#[derive(Clone, Copy, PartialEq)]
pub struct Point<X, Y>
{
    pub x : X,
    pub y : Y,
}
pub union AtomicPoint
{
    atomic : ManuallyDrop<AtomicU32>,
    simple : u32,
    split :  (u16, u16),
    point :  Point<u16, u16>,
}
assert_eq!(std::mem::size_of::<AtomicPoint>(), 4);
assert_eq!(std::mem::align_of::<AtomicPoint>(), 4);

pub struct SemiAtomicRect
{
    pub min :       AtomicPoint,
    pub max :       AtomicPoint,
    pub full_size : AtomicPoint,
    changed :       AtomicBool,
}

#[atomic_enum]
#[derive(PartialEq)]
pub enum TTViewState
{
    Valid,
    Processing,
    Changed,
    Invalid,
}
#[derive(Clone)]
pub struct Thermogram
{
    pub image :  TextureHandle,
    pub legend : TTGradients,
    pub scale :  [f64; 33],
}

#[derive(Clone, PartialEq, strum_macros::AsRefStr)]
pub enum TTGradients
{
    Linear,
    Phase,
}
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct WaveletViewParams
{
    pub scale :        RangedVal,
    pub time :         RangedVal,
    pub wavelet :      WaveletType,
    pub display_mode : ComplexResultMode,
    pub denoise :      bool,
}
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct TimeViewParams
{
    pub time :    RangedVal,
    pub denoise : bool,
}
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct FourierViewParams
{
    pub freq :         RangedVal,
    pub display_mode : ComplexResultMode,
    pub denoise :      bool,
}

#[derive(
    Clone,
    Copy,
    PartialEq,
    Debug,
    strum_macros::IntoStaticStr,
    strum_macros::AsRefStr,
    EnumVariantNames,
)]
#[strum(serialize_all = "title_case")]
pub enum TTViewParams
{
    WaveletView(WaveletViewParams),
    TimeView(TimeViewParams),
    FourierView(FourierViewParams),
}
pub use TTViewParams::*;

pub struct ExecutionTimeMeas
{
    last_time : Instant,
    #[cfg(feature = "time_meas")]
    writer :    BufWriter<File>,
}

//=======================================
//=====Traits & Trait Implementations====
//=======================================

pub trait EnumUpdate<T>
{
    fn update(&mut self, new_val : &str);
}

impl<T> EnumUpdate<T> for T
where
    T : VariantNames,
    T : AsRef<str>,
    T : FromStr,
    <T as FromStr>::Err : std::fmt::Debug,
{
    fn update(&mut self, new_val : &str) { *self = T::from_str(new_val).unwrap(); }
}
impl EnumUpdate<TTViewParams> for TTViewParams
{
    fn update(&mut self, new_val : &str)
    {
        let timeview : &'static str = Self::time_default().into();
        let waveletview : &'static str = Self::wavelet_default().into();
        let fourierview : &'static str = Self::fourier_default().into();
        let frames;
        let denoise;
        let display_mode;
        match self
        {
            FourierView(params) =>
            {
                frames = params.freq.max * 2; //TODO FIX
                display_mode = params.display_mode;
                denoise = params.denoise;
            }
            TimeView(params) =>
            {
                frames = params.time.max + 1;
                display_mode = Default::default();
                denoise = params.denoise;
            }
            WaveletView(params) =>
            {
                frames = params.scale.max;
                display_mode = params.display_mode;
                denoise = params.denoise;
            }
        };
        if timeview == new_val
        {
            *self = Self::time_frames(frames, denoise)
        }
        else if waveletview == new_val
        {
            *self = Self::wavelet_frames(frames, denoise, display_mode)
        }
        else if fourierview == new_val
        {
            *self = Self::fourier_frames(frames, denoise, display_mode)
        }
        else
        {
            unreachable!()
        }
    }
}

//=======================================
//============Implementations============
//=======================================

impl Default for RangedVal
{
    fn default() -> Self
    {
        Self {
            val : 0,
            min : 0,
            max : 100,
        }
    }
}
impl AtomicPoint
{
    ///`val`- (x,y)
    pub fn set(&self, x : u16, y : u16)
    {
        unsafe {
            self.atomic.store(
                AtomicPoint {
                    point : Point { x : x, y : y },
                }
                .simple,
                Ordering::Relaxed,
            );
        }
    }
    pub fn get(&self) -> Point<u16, u16>
    {
        unsafe {
            AtomicPoint {
                simple : self.atomic.load(Ordering::Relaxed),
            }
            .point
        }
    }
}
impl SemiAtomicRect
{
    pub fn new(min : (u16, u16), max : (u16, u16), full : (u16, u16)) -> Self
    {
        Self {
            min :       AtomicPoint { split : min },
            max :       AtomicPoint { split : max },
            full_size : AtomicPoint { split : full },
            changed :   AtomicBool::new(false),
        }
    }
    pub fn set_min(&self, x : u16, y : u16)
    {
        let (mut x, mut y) = (x, y);
        let full = self.full_size.get();
        x = x.clamp(0, full.x - 1);
        y = y.clamp(0, full.y - 1);
        let max = self.max.get();
        self.min.set(u16::min(x, max.x), u16::min(y, max.y));
        self.max.set(u16::max(x, max.x), u16::max(y, max.y));
    }
    pub fn set_max(&self, x : u16, y : u16)
    {
        let (mut x, mut y) = (x, y);
        let full = self.full_size.get();
        x = x.clamp(0, full.x - 1);
        y = y.clamp(0, full.y - 1);
        let min = self.min.get();
        self.min.set(u16::min(x, min.x), u16::min(y, min.y));
        self.max.set(u16::max(x, min.x), u16::max(y, min.y));
    }
    pub fn changed(&self, set : bool) -> bool { self.changed.swap(set, Ordering::Relaxed) }
}

impl Thermogram
{
    pub fn new(image : TextureHandle) -> Self
    {
        Self {
            image,
            legend : TTGradients::Linear,
            scale : [0.0; 33],
        }
    }
}

impl TTGradients
{
    pub fn raw_grad(&self) -> &colorgrad::Gradient
    {
        match self
        {
            TTGradients::Linear => &(*linear_grad),
            TTGradients::Phase => &(*phase_grad),
        }
    }
    pub fn grad_legend(&self) -> &TextureHandle
    {
        match self
        {
            TTGradients::Linear =>
            unsafe {
                match linear_grad_legend
                {
                    Some(ref inner) => inner,
                    None => unreachable!(),
                }
            },
            TTGradients::Phase =>
            unsafe {
                match phase_grad_legend
                {
                    Some(ref inner) => inner,
                    None => unreachable!(),
                }
            },
        }
    }

    pub fn init_grad(ctx : &Context)
    {
        unsafe {
            linear_grad_legend = Self::Linear.gen_legend(ctx);
            phase_grad_legend = Self::Phase.gen_legend(ctx);
        }
    }
    fn gen_legend(&self, ctx : &Context) -> Option<TextureHandle>
    {
        let mut legend = self
            .raw_grad()
            .colors(1024)
            .into_iter()
            .rev()
            .map(|color| {
                let color = color.to_rgba8();
                [
                    color[0], color[1], color[2], color[0], color[1], color[2], color[0], color[1],
                    color[2],
                ]
            })
            .flatten()
            .collect::<Vec<u8>>();
        let _major_bars_localizations = iter::once([0, 1, 2, 3, 4])
            .chain((1..8_usize).map(|x| {
                let x = x * 4 * 32;
                [x - 2, x - 1, x, x + 1, x + 2]
            }))
            .chain(iter::once([1019, 1020, 1021, 1022, 1023]))
            .flatten()
            .for_each(|x| {
                let x = x * 9;
                legend[x + 3] = 0;
                legend[x + 4] = 0;
                legend[x + 5] = 0;
                legend[x + 6] = 0;
                legend[x + 7] = 0;
                legend[x + 8] = 0;
            });
        let _normal_bars_localizations = (0..8_usize)
            .map(|x| {
                let x = (x * 4 + 2) * 32;
                [x - 2, x - 1, x, x + 1, x + 2]
            })
            .flatten()
            .for_each(|x| {
                let x = x * 9;
                legend[x + 6] = 0;
                legend[x + 7] = 0;
                legend[x + 8] = 0;
            });
        let _minor_bars_localizations = (0..8_usize)
            .map(|x| {
                let x = (x * 4 + 1) * 32;
                [x - 1, x, x + 1]
            })
            .chain((0..8_usize).map(|x| {
                let x = (x * 4 + 3) * 32;
                [x - 1, x, x + 1]
            }))
            .flatten()
            .for_each(|x| {
                let x = x * 9;
                legend[x + 6] = 0;
                legend[x + 7] = 0;
                legend[x + 8] = 0;
            });
        let legend = ColorImage::from_rgb([3, 1024], &legend);
        Some(ctx.load_texture(self.as_ref(), legend, TextureOptions::LINEAR))
    }
}

impl TTViewParams
{
    pub fn time_default() -> Self
    {
        TimeView(TimeViewParams {
            time :    Default::default(),
            denoise : true,
        })
    }
    pub fn fourier_default() -> Self
    {
        FourierView(FourierViewParams {
            freq :         Default::default(),
            display_mode : Default::default(),
            denoise :      true,
        })
    }
    pub fn wavelet_default() -> Self
    {
        WaveletView(WaveletViewParams {
            scale :        Default::default(),
            time :         Default::default(),
            wavelet :      Default::default(),
            display_mode : Default::default(),
            denoise :      true,
        })
    }
    pub fn time_frames(frames : usize, denoise : bool) -> Self
    {
        TimeView(TimeViewParams {
            time : RangedVal {
                val : 0,
                min : 0,
                max : frames - 1,
            },
            denoise,
        })
    }
    pub fn fourier_frames(frames : usize, denoise : bool, display_mode : ComplexResultMode)
        -> Self
    {
        FourierView(FourierViewParams {
            freq : RangedVal {
                val : 0,
                min : 0,
                max : frames / 2,
            },
            display_mode,
            denoise,
        })
    }
    pub fn wavelet_frames(frames : usize, denoise : bool, display_mode : ComplexResultMode)
        -> Self
    {
        WaveletView(WaveletViewParams {
            time : RangedVal {
                val : 0,
                min : 0,
                max : frames - 1,
            },
            scale : RangedVal {
                val : 1,
                min : 1,
                max : frames,
            },
            wavelet : Default::default(),
            display_mode,
            denoise,
        })
    }
    pub fn wavelet_wavelet(wavelet : WaveletType, mode : ComplexResultMode) -> Self
    {
        WaveletView(WaveletViewParams {
            time : Default::default(),
            scale : Default::default(),
            wavelet,
            display_mode : mode,
            denoise : true,
        })
    }
}

impl ExecutionTimeMeas
{
    pub fn new(_file_name : &str) -> Self
    {
        let temp = Instant::now();
        #[cfg(feature = "time_meas")]
        let f = File::create(_file_name).unwrap();
        Self {
            last_time : temp,

            #[cfg(feature = "time_meas")]
            writer :                               BufWriter::new(f),
        }
    }
    pub fn start(&mut self) { self.last_time = Instant::now(); }
    pub fn stop_print(&mut self, _text : &str) -> Duration
    {
        let duration = self.stop();
        #[cfg(feature = "time_meas")]
        {
            let _ = write!(self.writer, "{}: {:?}\n", _text, duration);
            let _ = self.writer.flush();
        }
        duration
    }
    pub fn stop(&mut self) -> Duration
    {
        let new_time = Instant::now();
        let duration = new_time.duration_since(self.last_time);
        self.last_time = new_time;
        duration
    }
}
//=======================================
//================Statics================
//=======================================

static mut phase_grad_legend : Option<TextureHandle> = None;
static mut linear_grad_legend : Option<TextureHandle> = None;
lazy_static! {
    static ref phase_grad: colorgrad::Gradient = colorgrad::CustomGradient::new()
        .colors(&[
            colorgrad::Color::new(0.975609, 0.520118, 0.970963, 1.0),
            colorgrad::Color::new(0.980325, 0.517215, 0.963189, 1.0),
            colorgrad::Color::new(0.984195, 0.513568, 0.954482, 1.0),
            colorgrad::Color::new(0.987253, 0.509216, 0.944879, 1.0),
            colorgrad::Color::new(0.989542, 0.504200, 0.934424, 1.0),
            colorgrad::Color::new(0.991117, 0.498555, 0.923199, 1.0),
            colorgrad::Color::new(0.992044, 0.492333, 0.911264, 1.0),
            colorgrad::Color::new(0.992386, 0.485608, 0.898711, 1.0),
            colorgrad::Color::new(0.992208, 0.478440, 0.885626, 1.0),
            colorgrad::Color::new(0.991576, 0.470885, 0.872092, 1.0),
            colorgrad::Color::new(0.990553, 0.462981, 0.858186, 1.0),
            colorgrad::Color::new(0.989194, 0.454827, 0.843992, 1.0),
            colorgrad::Color::new(0.987544, 0.446415, 0.829579, 1.0),
            colorgrad::Color::new(0.985648, 0.437830, 0.814990, 1.0),
            colorgrad::Color::new(0.983537, 0.429071, 0.800291, 1.0),
            colorgrad::Color::new(0.981245, 0.420188, 0.785511, 1.0),
            colorgrad::Color::new(0.978786, 0.411201, 0.770684, 1.0),
            colorgrad::Color::new(0.976182, 0.402116, 0.755835, 1.0),
            colorgrad::Color::new(0.973444, 0.392971, 0.740978, 1.0),
            colorgrad::Color::new(0.970571, 0.383765, 0.726138, 1.0),
            colorgrad::Color::new(0.967575, 0.374503, 0.711311, 1.0),
            colorgrad::Color::new(0.964440, 0.365189, 0.696498, 1.0),
            colorgrad::Color::new(0.961184, 0.355864, 0.681701, 1.0),
            colorgrad::Color::new(0.957791, 0.346503, 0.666936, 1.0),
            colorgrad::Color::new(0.954249, 0.337156, 0.652191, 1.0),
            colorgrad::Color::new(0.950551, 0.327802, 0.637451, 1.0),
            colorgrad::Color::new(0.946681, 0.318488, 0.622713, 1.0),
            colorgrad::Color::new(0.942637, 0.309226, 0.607977, 1.0),
            colorgrad::Color::new(0.938393, 0.299997, 0.593222, 1.0),
            colorgrad::Color::new(0.933944, 0.290868, 0.578444, 1.0),
            colorgrad::Color::new(0.929287, 0.281847, 0.563657, 1.0),
            colorgrad::Color::new(0.924409, 0.272939, 0.548831, 1.0),
            colorgrad::Color::new(0.919305, 0.264182, 0.533960, 1.0),
            colorgrad::Color::new(0.913979, 0.255566, 0.519077, 1.0),
            colorgrad::Color::new(0.908420, 0.247117, 0.504149, 1.0),
            colorgrad::Color::new(0.902640, 0.238833, 0.489176, 1.0),
            colorgrad::Color::new(0.896646, 0.230734, 0.474192, 1.0),
            colorgrad::Color::new(0.890441, 0.222811, 0.459182, 1.0),
            colorgrad::Color::new(0.884048, 0.215011, 0.444155, 1.0),
            colorgrad::Color::new(0.877467, 0.207399, 0.429109, 1.0),
            colorgrad::Color::new(0.870720, 0.199872, 0.414066, 1.0),
            colorgrad::Color::new(0.863829, 0.192502, 0.399025, 1.0),
            colorgrad::Color::new(0.856787, 0.185227, 0.384010, 1.0),
            colorgrad::Color::new(0.849624, 0.178009, 0.369000, 1.0),
            colorgrad::Color::new(0.842350, 0.170816, 0.354007, 1.0),
            colorgrad::Color::new(0.834978, 0.163729, 0.339075, 1.0),
            colorgrad::Color::new(0.827528, 0.156645, 0.324159, 1.0),
            colorgrad::Color::new(0.820013, 0.149618, 0.309330, 1.0),
            colorgrad::Color::new(0.812451, 0.142614, 0.294521, 1.0),
            colorgrad::Color::new(0.804858, 0.135667, 0.279821, 1.0),
            colorgrad::Color::new(0.797250, 0.128854, 0.265183, 1.0),
            colorgrad::Color::new(0.789662, 0.122170, 0.250661, 1.0),
            colorgrad::Color::new(0.782115, 0.115783, 0.236278, 1.0),
            colorgrad::Color::new(0.774642, 0.109705, 0.222009, 1.0),
            colorgrad::Color::new(0.767296, 0.104001, 0.207949, 1.0),
            colorgrad::Color::new(0.760123, 0.098945, 0.194118, 1.0),
            colorgrad::Color::new(0.753173, 0.094721, 0.180491, 1.0),
            colorgrad::Color::new(0.746518, 0.091421, 0.167231, 1.0),
            colorgrad::Color::new(0.740217, 0.089258, 0.154263, 1.0),
            colorgrad::Color::new(0.734335, 0.088445, 0.141736, 1.0),
            colorgrad::Color::new(0.728940, 0.089069, 0.129628, 1.0),
            colorgrad::Color::new(0.724103, 0.091195, 0.117995, 1.0),
            colorgrad::Color::new(0.719878, 0.094767, 0.106894, 1.0),
            colorgrad::Color::new(0.716312, 0.099636, 0.096236, 1.0),
            colorgrad::Color::new(0.713433, 0.105801, 0.086225, 1.0),
            colorgrad::Color::new(0.711275, 0.113024, 0.076774, 1.0),
            colorgrad::Color::new(0.709830, 0.121038, 0.068001, 1.0),
            colorgrad::Color::new(0.709098, 0.129894, 0.059810, 1.0),
            colorgrad::Color::new(0.709044, 0.139202, 0.052084, 1.0),
            colorgrad::Color::new(0.709630, 0.149017, 0.045170, 1.0),
            colorgrad::Color::new(0.710808, 0.159138, 0.038855, 1.0),
            colorgrad::Color::new(0.712514, 0.169499, 0.033292, 1.0),
            colorgrad::Color::new(0.714696, 0.179947, 0.028947, 1.0),
            colorgrad::Color::new(0.717291, 0.190504, 0.025470, 1.0),
            colorgrad::Color::new(0.720213, 0.201067, 0.022733, 1.0),
            colorgrad::Color::new(0.723417, 0.211657, 0.020622, 1.0),
            colorgrad::Color::new(0.726850, 0.222137, 0.019034, 1.0),
            colorgrad::Color::new(0.730457, 0.232565, 0.017876, 1.0),
            colorgrad::Color::new(0.734193, 0.242879, 0.017071, 1.0),
            colorgrad::Color::new(0.738018, 0.253114, 0.016547, 1.0),
            colorgrad::Color::new(0.741915, 0.263227, 0.016249, 1.0),
            colorgrad::Color::new(0.745847, 0.273221, 0.016125, 1.0),
            colorgrad::Color::new(0.749795, 0.283111, 0.016137, 1.0),
            colorgrad::Color::new(0.753751, 0.292884, 0.016250, 1.0),
            colorgrad::Color::new(0.757692, 0.302570, 0.016440, 1.0),
            colorgrad::Color::new(0.761605, 0.312172, 0.016684, 1.0),
            colorgrad::Color::new(0.765495, 0.321712, 0.016989, 1.0),
            colorgrad::Color::new(0.769322, 0.331165, 0.017296, 1.0),
            colorgrad::Color::new(0.773097, 0.340580, 0.017600, 1.0),
            colorgrad::Color::new(0.776812, 0.349937, 0.017896, 1.0),
            colorgrad::Color::new(0.780442, 0.359250, 0.018177, 1.0),
            colorgrad::Color::new(0.784000, 0.368555, 0.018435, 1.0),
            colorgrad::Color::new(0.787459, 0.377827, 0.018665, 1.0),
            colorgrad::Color::new(0.790825, 0.387090, 0.018861, 1.0),
            colorgrad::Color::new(0.794094, 0.396362, 0.019014, 1.0),
            colorgrad::Color::new(0.797244, 0.405638, 0.019122, 1.0),
            colorgrad::Color::new(0.800294, 0.414903, 0.019179, 1.0),
            colorgrad::Color::new(0.803223, 0.424179, 0.019183, 1.0),
            colorgrad::Color::new(0.806046, 0.433460, 0.019132, 1.0),
            colorgrad::Color::new(0.808758, 0.442759, 0.019026, 1.0),
            colorgrad::Color::new(0.811360, 0.452053, 0.018866, 1.0),
            colorgrad::Color::new(0.813856, 0.461340, 0.018654, 1.0),
            colorgrad::Color::new(0.816265, 0.470631, 0.018394, 1.0),
            colorgrad::Color::new(0.818571, 0.479917, 0.018090, 1.0),
            colorgrad::Color::new(0.820801, 0.489183, 0.017747, 1.0),
            colorgrad::Color::new(0.822954, 0.498459, 0.017369, 1.0),
            colorgrad::Color::new(0.825032, 0.507709, 0.016963, 1.0),
            colorgrad::Color::new(0.827042, 0.516942, 0.016575, 1.0),
            colorgrad::Color::new(0.828978, 0.526150, 0.016225, 1.0),
            colorgrad::Color::new(0.830837, 0.535317, 0.015949, 1.0),
            colorgrad::Color::new(0.832616, 0.544456, 0.015792, 1.0),
            colorgrad::Color::new(0.834313, 0.553536, 0.015820, 1.0),
            colorgrad::Color::new(0.835903, 0.562570, 0.016118, 1.0),
            colorgrad::Color::new(0.837376, 0.571505, 0.016798, 1.0),
            colorgrad::Color::new(0.838696, 0.580351, 0.018004, 1.0),
            colorgrad::Color::new(0.839847, 0.589056, 0.019918, 1.0),
            colorgrad::Color::new(0.840771, 0.597613, 0.022763, 1.0),
            colorgrad::Color::new(0.841431, 0.605951, 0.026810, 1.0),
            colorgrad::Color::new(0.841773, 0.614040, 0.032376, 1.0),
            colorgrad::Color::new(0.841735, 0.621817, 0.040011, 1.0),
            colorgrad::Color::new(0.841252, 0.629224, 0.049120, 1.0),
            colorgrad::Color::new(0.840261, 0.636189, 0.059305, 1.0),
            colorgrad::Color::new(0.838684, 0.642636, 0.070471, 1.0),
            colorgrad::Color::new(0.836472, 0.648502, 0.082640, 1.0),
            colorgrad::Color::new(0.833559, 0.653724, 0.095596, 1.0),
            colorgrad::Color::new(0.829894, 0.658244, 0.109262, 1.0),
            colorgrad::Color::new(0.825436, 0.662013, 0.123432, 1.0),
            colorgrad::Color::new(0.820169, 0.664981, 0.138185, 1.0),
            colorgrad::Color::new(0.814079, 0.667153, 0.153209, 1.0),
            colorgrad::Color::new(0.807174, 0.668516, 0.168557, 1.0),
            colorgrad::Color::new(0.799467, 0.669078, 0.184047, 1.0),
            colorgrad::Color::new(0.790984, 0.668874, 0.199610, 1.0),
            colorgrad::Color::new(0.781782, 0.667950, 0.215201, 1.0),
            colorgrad::Color::new(0.771898, 0.666350, 0.230722, 1.0),
            colorgrad::Color::new(0.761389, 0.664160, 0.246105, 1.0),
            colorgrad::Color::new(0.750316, 0.661435, 0.261333, 1.0),
            colorgrad::Color::new(0.738731, 0.658248, 0.276357, 1.0),
            colorgrad::Color::new(0.726679, 0.654667, 0.291131, 1.0),
            colorgrad::Color::new(0.714212, 0.650774, 0.305673, 1.0),
            colorgrad::Color::new(0.701369, 0.646617, 0.319948, 1.0),
            colorgrad::Color::new(0.688182, 0.642255, 0.333971, 1.0),
            colorgrad::Color::new(0.674664, 0.637733, 0.347744, 1.0),
            colorgrad::Color::new(0.660836, 0.633096, 0.361267, 1.0),
            colorgrad::Color::new(0.646712, 0.628377, 0.374577, 1.0),
            colorgrad::Color::new(0.632276, 0.623598, 0.387655, 1.0),
            colorgrad::Color::new(0.617525, 0.618782, 0.400558, 1.0),
            colorgrad::Color::new(0.602450, 0.613930, 0.413255, 1.0),
            colorgrad::Color::new(0.587033, 0.609090, 0.425781, 1.0),
            colorgrad::Color::new(0.571260, 0.604231, 0.438168, 1.0),
            colorgrad::Color::new(0.555110, 0.599363, 0.450404, 1.0),
            colorgrad::Color::new(0.538576, 0.594493, 0.462504, 1.0),
            colorgrad::Color::new(0.521599, 0.589621, 0.474508, 1.0),
            colorgrad::Color::new(0.504201, 0.584753, 0.486396, 1.0),
            colorgrad::Color::new(0.486316, 0.579863, 0.498214, 1.0),
            colorgrad::Color::new(0.467958, 0.574931, 0.509934, 1.0),
            colorgrad::Color::new(0.449120, 0.569964, 0.521598, 1.0),
            colorgrad::Color::new(0.429782, 0.564934, 0.533218, 1.0),
            colorgrad::Color::new(0.409982, 0.559832, 0.544799, 1.0),
            colorgrad::Color::new(0.389720, 0.554613, 0.556346, 1.0),
            colorgrad::Color::new(0.369082, 0.549264, 0.567890, 1.0),
            colorgrad::Color::new(0.348105, 0.543747, 0.579434, 1.0),
            colorgrad::Color::new(0.326877, 0.538047, 0.590979, 1.0),
            colorgrad::Color::new(0.305561, 0.532112, 0.602556, 1.0),
            colorgrad::Color::new(0.284236, 0.525924, 0.614165, 1.0),
            colorgrad::Color::new(0.263159, 0.519446, 0.625830, 1.0),
            colorgrad::Color::new(0.242495, 0.512650, 0.637551, 1.0),
            colorgrad::Color::new(0.222572, 0.505507, 0.649339, 1.0),
            colorgrad::Color::new(0.203679, 0.498022, 0.661190, 1.0),
            colorgrad::Color::new(0.186189, 0.490122, 0.673126, 1.0),
            colorgrad::Color::new(0.170514, 0.481834, 0.685136, 1.0),
            colorgrad::Color::new(0.157117, 0.473118, 0.697232, 1.0),
            colorgrad::Color::new(0.146378, 0.464018, 0.709410, 1.0),
            colorgrad::Color::new(0.138585, 0.454497, 0.721643, 1.0),
            colorgrad::Color::new(0.133804, 0.444580, 0.733949, 1.0),
            colorgrad::Color::new(0.131904, 0.434263, 0.746297, 1.0),
            colorgrad::Color::new(0.132531, 0.423594, 0.758686, 1.0),
            colorgrad::Color::new(0.135191, 0.412602, 0.771094, 1.0),
            colorgrad::Color::new(0.139304, 0.401284, 0.783498, 1.0),
            colorgrad::Color::new(0.144495, 0.389694, 0.795878, 1.0),
            colorgrad::Color::new(0.150242, 0.377897, 0.808211, 1.0),
            colorgrad::Color::new(0.156303, 0.365938, 0.820451, 1.0),
            colorgrad::Color::new(0.162459, 0.353867, 0.832570, 1.0),
            colorgrad::Color::new(0.168638, 0.341794, 0.844516, 1.0),
            colorgrad::Color::new(0.174839, 0.329789, 0.856233, 1.0),
            colorgrad::Color::new(0.181049, 0.317937, 0.867668, 1.0),
            colorgrad::Color::new(0.187461, 0.306394, 0.878752, 1.0),
            colorgrad::Color::new(0.194078, 0.295237, 0.889426, 1.0),
            colorgrad::Color::new(0.201013, 0.284653, 0.899619, 1.0),
            colorgrad::Color::new(0.208478, 0.274783, 0.909260, 1.0),
            colorgrad::Color::new(0.216496, 0.265785, 0.918299, 1.0),
            colorgrad::Color::new(0.225063, 0.257783, 0.926677, 1.0),
            colorgrad::Color::new(0.234299, 0.250981, 0.934357, 1.0),
            colorgrad::Color::new(0.244088, 0.245443, 0.941312, 1.0),
            colorgrad::Color::new(0.254476, 0.241339, 0.947531, 1.0),
            colorgrad::Color::new(0.265294, 0.238660, 0.953012, 1.0),
            colorgrad::Color::new(0.276520, 0.237496, 0.957776, 1.0),
            colorgrad::Color::new(0.287984, 0.237734, 0.961852, 1.0),
            colorgrad::Color::new(0.299643, 0.239327, 0.965296, 1.0),
            colorgrad::Color::new(0.311348, 0.242176, 0.968161, 1.0),
            colorgrad::Color::new(0.323045, 0.246128, 0.970496, 1.0),
            colorgrad::Color::new(0.334642, 0.251060, 0.972381, 1.0),
            colorgrad::Color::new(0.346059, 0.256764, 0.973864, 1.0),
            colorgrad::Color::new(0.357329, 0.263129, 0.975020, 1.0),
            colorgrad::Color::new(0.368354, 0.269998, 0.975904, 1.0),
            colorgrad::Color::new(0.379155, 0.277244, 0.976570, 1.0),
            colorgrad::Color::new(0.389731, 0.284764, 0.977064, 1.0),
            colorgrad::Color::new(0.400095, 0.292503, 0.977423, 1.0),
            colorgrad::Color::new(0.410262, 0.300363, 0.977682, 1.0),
            colorgrad::Color::new(0.420229, 0.308302, 0.977868, 1.0),
            colorgrad::Color::new(0.430058, 0.316272, 0.978004, 1.0),
            colorgrad::Color::new(0.439768, 0.324207, 0.978110, 1.0),
            colorgrad::Color::new(0.449401, 0.332129, 0.978200, 1.0),
            colorgrad::Color::new(0.458996, 0.339971, 0.978287, 1.0),
            colorgrad::Color::new(0.468590, 0.347722, 0.978377, 1.0),
            colorgrad::Color::new(0.478245, 0.355355, 0.978492, 1.0),
            colorgrad::Color::new(0.488019, 0.362824, 0.978640, 1.0),
            colorgrad::Color::new(0.497946, 0.370147, 0.978828, 1.0),
            colorgrad::Color::new(0.508073, 0.377305, 0.979061, 1.0),
            colorgrad::Color::new(0.518451, 0.384258, 0.979347, 1.0),
            colorgrad::Color::new(0.529120, 0.391021, 0.979691, 1.0),
            colorgrad::Color::new(0.540096, 0.397551, 0.980097, 1.0),
            colorgrad::Color::new(0.551427, 0.403865, 0.980569, 1.0),
            colorgrad::Color::new(0.563108, 0.409948, 0.981113, 1.0),
            colorgrad::Color::new(0.575134, 0.415789, 0.981731, 1.0),
            colorgrad::Color::new(0.587510, 0.421405, 0.982418, 1.0),
            colorgrad::Color::new(0.600220, 0.426814, 0.983172, 1.0),
            colorgrad::Color::new(0.613223, 0.432013, 0.983998, 1.0),
            colorgrad::Color::new(0.626503, 0.437003, 0.984886, 1.0),
            colorgrad::Color::new(0.640016, 0.441823, 0.985827, 1.0),
            colorgrad::Color::new(0.653712, 0.446473, 0.986823, 1.0),
            colorgrad::Color::new(0.667569, 0.450995, 0.987859, 1.0),
            colorgrad::Color::new(0.681527, 0.455396, 0.988937, 1.0),
            colorgrad::Color::new(0.695577, 0.459695, 0.990045, 1.0),
            colorgrad::Color::new(0.709665, 0.463920, 0.991178, 1.0),
            colorgrad::Color::new(0.723765, 0.468069, 0.992333, 1.0),
            colorgrad::Color::new(0.737851, 0.472183, 0.993503, 1.0),
            colorgrad::Color::new(0.751904, 0.476241, 0.994667, 1.0),
            colorgrad::Color::new(0.765890, 0.480264, 0.995824, 1.0),
            colorgrad::Color::new(0.779793, 0.484245, 0.996963, 1.0),
            colorgrad::Color::new(0.793606, 0.488206, 0.998069, 1.0),
            colorgrad::Color::new(0.807288, 0.492107, 0.999125, 1.0),
            colorgrad::Color::new(0.820827, 0.495947, 1.000000, 1.0),
            colorgrad::Color::new(0.834201, 0.499717, 1.000000, 1.0),
            colorgrad::Color::new(0.847361, 0.503390, 1.000000, 1.0),
            colorgrad::Color::new(0.860287, 0.506933, 1.000000, 1.0),
            colorgrad::Color::new(0.872915, 0.510299, 1.000000, 1.0),
            colorgrad::Color::new(0.885202, 0.513454, 1.000000, 1.0),
            colorgrad::Color::new(0.897090, 0.516325, 1.000000, 1.0),
            colorgrad::Color::new(0.908507, 0.518885, 1.000000, 1.0),
            colorgrad::Color::new(0.919391, 0.521035, 1.000000, 1.0),
            colorgrad::Color::new(0.929679, 0.522735, 0.998520, 1.0),
            colorgrad::Color::new(0.939301, 0.523911, 0.995999, 1.0),
            colorgrad::Color::new(0.948193, 0.524505, 0.992736, 1.0),
            colorgrad::Color::new(0.956308, 0.524462, 0.988644, 1.0),
            colorgrad::Color::new(0.963596, 0.523740, 0.983677, 1.0),
            colorgrad::Color::new(0.970035, 0.522293, 0.977792, 1.0),
        ])
        .interpolation(colorgrad::Interpolation::Linear)
        .build()
        .unwrap();
    static ref linear_grad: colorgrad::Gradient = colorgrad::inferno();
}
