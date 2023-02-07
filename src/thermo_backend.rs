use egui::mutex::Mutex;
use egui_extras::RetainedImage;
use ndarray::Array2;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar};
use std::thread::{self, JoinHandle};
use tribuf::TripleBuffer;
use triple_buffer as tribuf;
use triple_buffer::triple_buffer;

#[path = "tt_input_data.rs"]
mod tt_input_data;
#[derive(Clone, Copy)]
pub enum WtResultMode
{
    Phase,
    Magnitude,
    Real,
    Imaginary,
}
#[derive(Clone, Copy)]
pub enum WaveletType
{
    Morlet,
}
#[derive(Clone, Copy)]
pub struct RangedVal
{
    val : usize,
    min : usize,
    max : usize,
}
impl Default for RangedVal
{
    fn default() -> Self
    {
        Self {
            val : 0,
            min : 0,
            max : 1,
        }
    }
}
#[derive(Clone, Copy)]
pub enum TTViewParams
{
    TransformView
    {
        scale :   RangedVal,
        time :    RangedVal,
        wavelet : WaveletType,
        mode :    WtResultMode,
    },
    TimeView
    {
        time : RangedVal
    },
}

use TTViewParams::*;
#[derive(Clone, Copy)]
pub enum TTViewState
{
    Valid(TTViewParams),
    Processing(TTViewParams),
    Invalid(TTViewParams),
    Error,
}

pub struct TTViewBackend
{
    state : tribuf::Output<TTViewState>,
    image : tribuf::Input<Option<RetainedImage>>,
}

pub struct TTViewGUI
{
    state : tribuf::Input<TTViewState>,
    image : tribuf::Output<Option<RetainedImage>>,
}
fn TTView_new(state : TTViewState) -> (TTViewGUI, TTViewBackend)
{
    let (state_input, state_output) : (tribuf::Input<TTViewState>, tribuf::Output<TTViewState>) =
        TripleBuffer::new(&state).split();
    let (image_input, image_output) : (
        tribuf::Input<Option<RetainedImage>>,
        tribuf::Output<Option<RetainedImage>>,
    ) = TripleBuffer::default().split();
    (
        TTViewGUI {
            state : state_input,
            image : image_output,
        },
        TTViewBackend {
            state : state_output,
            image : image_input,
        },
    )
}
pub struct TTFileBackend
{
    width :      Arc<AtomicUsize>,
    height :     Arc<AtomicUsize>,
    frames :     Arc<AtomicUsize>,
    path :       tribuf::Output<Option<PathBuf>>,
    input_data : Option<tt_input_data::TTInputData>,
}
pub struct TTFileGUI
{
    width :  Arc<AtomicUsize>,
    height : Arc<AtomicUsize>,
    frames : Arc<AtomicUsize>,
    path :   tribuf::Input<Option<PathBuf>>,
}
pub struct TTStateBackend
{
    views :     [TTViewBackend; 4],
    changed :   Arc<(Mutex<bool>, Condvar)>,
    stop_flag : Arc<AtomicBool>,
    file :      TTFileBackend,
}
impl TTStateBackend
{
    fn run(&mut self) -> ()
    {
        while self.stop_flag.load(Ordering::Relaxed) == false
        {}
    }
}
pub struct TTStateGUI
{
    views :     [TTViewGUI; 4],
    changed :   Arc<(Mutex<bool>, Condvar)>,
    stop_flag : Arc<AtomicBool>,
    file :      TTFileGUI,
}
pub struct TTState
{
    state :          TTStateGUI,
    backend_handle : Option<JoinHandle<()>>,
}

impl TTState
{
    pub fn new() -> Self
    {
        let view_states = [
            TTViewState::Invalid(TimeView {
                time : RangedVal::default(),
            }),
            TTViewState::Invalid(TransformView {
                scale :   RangedVal::default(),
                time :    RangedVal::default(),
                wavelet : WaveletType::Morlet,
                mode :    WtResultMode::Phase,
            }),
            TTViewState::Invalid(TransformView {
                scale :   RangedVal::default(),
                time :    RangedVal::default(),
                wavelet : WaveletType::Morlet,
                mode :    WtResultMode::Magnitude,
            }),
            TTViewState::Invalid(TimeView {
                time : RangedVal::default(),
            }),
        ];
        // let (mut views_gui, mut views_backend) : ([TTViewGUI; 4], [TTViewBackend; 4]) =
        let [(g1,b1),(g2,b2),(g3,b3),(g4,b4)] //: [(TTViewGUI, TTViewBackend); 4] 
        = view_states.map( |x| TTView_new(x));

        let (path_gui, path_backend) = triple_buffer(&None);

        let gui = TTStateGUI {
            views :     [g1, g2, g3, g4],
            changed :   Arc::new((Mutex::new(false), Condvar::new())),
            stop_flag : Arc::new(AtomicBool::new(false)),
            file :      TTFileGUI {
                width :  Arc::new(AtomicUsize::new(0)),
                height : Arc::new(AtomicUsize::new(0)),
                frames : Arc::new(AtomicUsize::new(0)),
                path :   path_gui,
            },
        };
        let backend = TTStateBackend {
            views :     [b1, b2, b3, b4],
            changed :   gui.changed.clone(),
            stop_flag : gui.stop_flag.clone(),
            file :      TTFileBackend {
                width :      gui.file.width.clone(),
                height :     gui.file.height.clone(),
                frames :     gui.file.frames.clone(),
                path :       path_backend,
                input_data : None,
            },
        };

        Self {
            state :          gui,
            backend_handle : Some(thread::spawn(move || {
                let mut backend_state = backend;
                backend_state.run();
            })),
        }
    }
}
impl Drop for TTState
{
    fn drop(&mut self)
    {
        self.state.stop_flag.store(true, Ordering::SeqCst);
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
