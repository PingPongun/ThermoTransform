use atomic_enum::atomic_enum;
use mathru::analysis::interpolation::spline::*;
use num_complex::Complex64;
use rayon::prelude::{
    IndexedParallelIterator,
    IntoParallelIterator,
    IntoParallelRefMutIterator,
    ParallelIterator,
};
use std::collections::HashMap;
use std::f64::consts::PI;
use strum_macros::{EnumString, EnumVariantNames};

use crate::tt_common::*;
// struct Param
// {
//     name :    String,
//     min_max : RangedVal,
// }
/// psi(wavelet function, d0psi)
/// d1psi-> first derivate of psi, d2psi-> second derivate...
/// for dxpsi fields(x is index of outermost dim-> d0psi, d1psi, d2psi ) are values these functions for 2 most extreme time values
#[derive(Clone, PartialEq)]
pub struct PolyWiseWavelet
{
    pub dxpsi : [Point<isize, [f64; 3]>; 2],
    pub d3psi : Vec<Point<isize, f64>>,
    pub psi :   Vec<f64>,
}
#[inline(always)]
fn onesided_extremals_search(
    extvec : &mut Vec<usize>,
    last_extremal_x : usize,
    mid_x : usize,
    y : &[f64],
    threshold : f64,
    fn_dist_btw_extremals : impl Fn(usize, usize) -> usize,
    fn_fake_extremal : impl Fn(usize, usize) -> usize,
    iter : impl Iterator<Item = usize>,
)
{
    let mut prev_convex = (y[mid_x + 1] - y[mid_x]) < (y[mid_x] - y[mid_x - 1]);
    let mut prev_concave = (y[mid_x + 1] - y[mid_x]) > (y[mid_x] - y[mid_x - 1]);
    let mut dist_btw_extremals = usize::MAX >> 1;
    let mut last_extremal_x = last_extremal_x;
    let mut below_thresh = false;

    for i in iter
    {
        let dyp = y[i + 1] - y[i];
        let dym = y[i] - y[i - 1];
        let convex = dyp < dym;
        let concave = dyp > dym;
        let mut extr = false;
        //check for extremes
        if (y[i + 1] < y[i] && y[i - 1] < y[i])
            || (y[i + 1] > y[i] && y[i - 1] > y[i])
            || (concave && prev_convex)
            || (convex && prev_concave)
        {
            extvec.push(i);
            dist_btw_extremals = fn_dist_btw_extremals(i, last_extremal_x);
            last_extremal_x = i;
            extr = true;
        }

        if fn_dist_btw_extremals(i, last_extremal_x) >> 1 > dist_btw_extremals
        {
            //if wavelet is i.e. exponetial(Poisson), there are no extremals, so to obtain correct aproximation we need to add some
            extvec.push(fn_fake_extremal(last_extremal_x, dist_btw_extremals));
            last_extremal_x = fn_fake_extremal(last_extremal_x, dist_btw_extremals);
            extr = true;
        }

        if extr
        {
            if y[*extvec.last().unwrap()].abs() < threshold
            {
                if below_thresh && extvec.len() > 5
                {
                    //stop iteration if two consecutive extremals are below threshold and ther has been found at least five extremals
                    break;
                }
                else
                {
                    below_thresh = true;
                }
            }
            else
            {
                below_thresh = false;
            }
        }
        prev_concave = concave;
        prev_convex = convex;
    }
}
impl PolyWiseWavelet
{
    pub fn new(x : &[isize], y : &[f64], init_calc_half_len : usize) -> Self
    {
        let glob_max = y.iter().cloned().reduce(|a, b| f64::max(a, b)).unwrap();
        let mid_x = x.len() >> 1;
        let threshold = glob_max * 0.025;

        let mut extremals_positive = Vec::with_capacity(init_calc_half_len);
        let mut extremals = Vec::with_capacity(init_calc_half_len);

        onesided_extremals_search(
            &mut extremals_positive,
            0,
            mid_x,
            y,
            threshold,
            |a, b| a - b,
            |a, b| a + b,
            (mid_x..x.len() - 1).into_iter(),
        );
        onesided_extremals_search(
            &mut extremals,
            usize::MAX,
            mid_x,
            y,
            threshold,
            |a, b| b - a,
            |a, b| a - b,
            (1..=mid_x).into_iter().rev(),
        );

        if extremals[0] == extremals_positive[0]
        {
            extremals.reverse();
            extremals.extend_from_slice(&extremals_positive[1..]);
        }
        else
        {
            extremals.reverse();
            extremals.extend_from_slice(&extremals_positive[..]);
        };
        let psi = y[*extremals.first().unwrap()..=*extremals.last().unwrap()].to_vec();
        let (xe, ye) : (Vec<f64>, Vec<f64>) =
            extremals.into_iter().map(|i| (x[i] as f64, y[i])).unzip();
        //interpolate wavelet function using extremal points
        let cubic_spline = CubicSpline::interpolate(&xe, &ye, CubicSplineConstraint::Natural);
        let cubic_spline_d1 = cubic_spline.differentiate();
        let cubic_spline_d2 = cubic_spline_d1.differentiate();
        let cubic_spline_d3 = cubic_spline_d2.differentiate();
        //`cubic_spline_d3` is picewise constant, with discontinuities at `x`
        //mathru implementation of cubic spline at this points evaluates to right-side value (x => x+)
        //  (exept last x which evals to left-side val (x_last=> x_last-))
        let (x0f, xJf) = (*xe.first().unwrap(), *xe.last().unwrap());
        let (x0, xJ) = (x0f as isize, xJf as isize);
        let mut d3psi_raw : Vec<_> = xe
            .iter()
            .map(|&x| {
                Point {
                    x : x as isize,
                    y : 6.0 * cubic_spline_d3.eval(x), //last two points have the same value (see above)
                }
            })
            .collect();
        let d3psi_raw_first = d3psi_raw.first().unwrap().clone();
        let mut d3psi_raw_last = d3psi_raw.pop().unwrap(); //remove last elem from `d3psi_raw`
        d3psi_raw_last.y *= -1.0;
        PolyWiseWavelet {
            //psi(x0), d1psi(xJ), d2psi(x0) are negated (its their negation, taht is needed for cwt calculations)
            //d3psi(i) is actually 6*d3psi(i)-6*d3psi(i-1) (& 6*d3psi(0) for i=0 & -6*d3psi(J) for i=J)
            dxpsi : [
                Point {
                    x : x0,
                    y : [
                        -cubic_spline.eval(x0f),
                        cubic_spline_d1.eval(x0f),
                        -cubic_spline_d2.eval(x0f),
                    ],
                },
                Point {
                    x : xJ,
                    y : [
                        cubic_spline.eval(xJf),
                        -cubic_spline_d1.eval(xJf),
                        cubic_spline_d2.eval(xJf),
                    ],
                },
            ],
            d3psi : [d3psi_raw_first]
                .into_iter()
                .chain(
                    d3psi_raw
                        .windows(2)
                        .map(|p| {
                            Point {
                                x : p[1].x,
                                y : p[1].y - p[0].y,
                            }
                        })
                        .chain([d3psi_raw_last].into_iter()),
                )
                .collect(),
            psi,
        }
    }
}

/// self.0[0]-> real part; self.0[1]-> img. part
#[derive(Clone, PartialEq)]
pub struct PolyWiseComplex(pub [PolyWiseWavelet; 2]);
impl PolyWiseComplex
{
    pub fn new(func : &dyn Fn(&f64, &f64) -> (f64, f64), scale : usize, time_step : f64) -> Self
    {
        let init_calc_half_len = 20 + scale * 6;
        let x_iter = (-(init_calc_half_len as isize)..=(init_calc_half_len as isize)).into_iter();
        let x_isize : Vec<isize> = x_iter.clone().collect();
        let x : Vec<f64> = x_iter.map(|x| (x as f64) * time_step).collect();
        let (yreal, yimg) : (Vec<f64>, Vec<f64>) =
            x.iter().map(|x| func(&x, &(scale as f64))).unzip();
        Self([
            PolyWiseWavelet::new(&x_isize, &yreal, init_calc_half_len),
            PolyWiseWavelet::new(&x_isize, &yimg, init_calc_half_len),
        ])
    }
}
pub struct WaveletParams
{
    pub func :     &'static (dyn Send + Sync + Fn(&f64, &f64) -> (f64, f64)), //aka. psi; Fn(t:time, s1:scale1)//, s2:scale2)
    // s1 :       Param,
    // s2 :       Option<Param>,
    pub wavelets : Vec<Option<PolyWiseComplex>>,
}

impl WaveletParams
{
    pub fn get_poly_wise(&mut self, scale : usize) -> PolyWiseComplex
    {
        if self.wavelets.len() <= scale
        {
            self.wavelets.resize(scale + 1, None);
        }
        if let Some(poly_wise) = self.wavelets[scale].clone()
        {
            return poly_wise;
        }
        else
        {
            let poly_wise = PolyWiseComplex::new(self.func, scale + 1, 1.0);
            self.wavelets[scale] = Some(poly_wise.clone());
            return poly_wise;
        }
    }
    pub fn uget_poly_wise(&self, scale : usize) -> PolyWiseComplex
    {
        self.wavelets[scale].clone().unwrap()
    }
    pub fn batch_calc(&mut self, scale : usize)
    {
        if self.wavelets.len() <= scale
        {
            self.wavelets.resize(scale + 1, None);
        }

        self.wavelets
            .par_iter_mut()
            .zip((0..scale).into_par_iter())
            .for_each(|(w, s)| {
                if None == *w
                {
                    let poly_wise = PolyWiseComplex::new(self.func, s + 1, 1.0);
                    *w = Some(poly_wise.clone());
                }
            })
    }
}

#[atomic_enum]
#[derive(PartialEq, Eq, Hash, Default, strum_macros::AsRefStr, EnumString, EnumVariantNames)]
#[strum(serialize_all = "title_case")]
pub enum WaveletType
{
    #[default]
    Morlet,
    Shannon,
    Modified_Shannon,
    BSpline_2,
    Poisson_1,
    Poisson_2,
}
use WaveletType::*;
// pub struct WaveletBank(HashMap<WaveletType, WaveletParams>);
pub type WaveletBank = HashMap<WaveletType, WaveletParams>;
pub trait WaveletBankTrait
{
    fn new_wb() -> Self;
}
impl WaveletBankTrait for WaveletBank
{
    fn new_wb() -> Self
    {
        HashMap::from([
            (
                Morlet,
                WaveletParams {
                    func :     &morlet_wavelet_func,
                    wavelets : Vec::new(),
                },
            ),
            (
                Shannon,
                WaveletParams {
                    func :     &shannon_wavelet_func,
                    wavelets : Vec::new(),
                },
            ),
            (
                Modified_Shannon,
                WaveletParams {
                    func :     &shannon_mod_wavelet_func,
                    wavelets : Vec::new(),
                },
            ),
            (
                BSpline_2,
                WaveletParams {
                    func :     &b_spline_wavelet_func_2,
                    wavelets : Vec::new(),
                },
            ),
            (
                Poisson_1,
                WaveletParams {
                    func :     &poisson_wavelet_func_1,
                    wavelets : Vec::new(),
                },
            ),
            (
                Poisson_2,
                WaveletParams {
                    func :     &poisson_wavelet_func_2,
                    wavelets : Vec::new(),
                },
            ),
        ])
    }
}

fn morlet_wavelet_func(t : &f64, s : &f64) -> (f64, f64)
{
    let ret = Complex64::new(
        (2.0 / (PI * s).sqrt()) * (-4.0 * (t / s).powi(2)).exp(),
        0.0,
    ) * Complex64::new(0.0, 8.0 * t / s).exp();
    (ret.re, ret.im)
}
#[inline(always)]
fn b_spline_wavelet_func_inner(t : &f64, s : &f64, p : &i32, fb : &f64) -> (f64, f64)
{
    let fbt = fb * t / s;
    let sinc = if fbt == 0. { 1. } else { fbt.sin() / fbt };
    let ret = Complex64::new((0.5 / (s).sqrt()) * sinc.powi(*p), 0.0)
        * Complex64::new(0.0, 4.0 * t / s).exp();
    (ret.re, ret.im)
}

fn b_spline_wavelet_func_2(t : &f64, s : &f64) -> (f64, f64)
{
    b_spline_wavelet_func_inner(t, s, &2, &4.)
}
fn shannon_wavelet_func(t : &f64, s : &f64) -> (f64, f64)
{
    b_spline_wavelet_func_inner(t, s, &1, &8.)
}
fn shannon_mod_wavelet_func(t : &f64, s : &f64) -> (f64, f64)
{
    let shannon = shannon_wavelet_func(t, s);
    let gauss = (-0.1 * (t / s).powi(2)).exp();
    (shannon.0 * gauss, shannon.1 * gauss)
}
#[inline(always)]
fn poisson_wavelet_func_inner(t : &f64, s : &f64, m : &i32, sc : &f64) -> (f64, f64)
{
    let a = 0.5 / PI / (s).sqrt();
    let ret = a * Complex64::new(1.0, -sc * t / s).powi(-1 - *m);
    (ret.re, ret.im)
}

fn poisson_wavelet_func_1(t : &f64, s : &f64) -> (f64, f64)
{
    poisson_wavelet_func_inner(t, s, &1, &4.)
}

fn poisson_wavelet_func_2(t : &f64, s : &f64) -> (f64, f64)
{
    poisson_wavelet_func_inner(t, s, &2, &2.)
}
