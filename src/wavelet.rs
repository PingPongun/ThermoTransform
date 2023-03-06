use mathru::analysis::interpolation::spline::*;
use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;
use strum_macros::{EnumString, EnumVariantNames};
///(x,y)
#[derive(Clone, PartialEq)]
pub struct Point<T>
{
    pub x : isize,
    pub y : T,
}
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
    pub dxpsi : [Point<[f64; 3]>; 2],
    pub d3psi : Vec<Point<f64>>,
}
impl PolyWiseWavelet
{
    pub fn new(x : &[isize], y : &[f64], init_calc_half_len : usize) -> Self
    {
        // let mut roots = Vec::with_capacity(init_calc_half_len);
        let mut extremals = Vec::with_capacity(init_calc_half_len);
        let mut glob_max = f64::NEG_INFINITY;
        let mut prev_convex = (y[2] - y[1]) < (y[1] - y[0]);
        let mut prev_concave = (y[2] - y[1]) > (y[1] - y[0]);
        //check for roots
        // if y[0] * y[1] <= 0.0
        // {
        //     if y[0].abs() < y[1].abs()
        //     {
        //         roots.push(0);
        //     }
        //     else
        //     {
        //         roots.push(1);
        //     }
        // }
        for i in 1..x.len() - 1
        {
            // //check for roots
            // if y[i + 1] * y[i] <= 0.0
            // {
            //     if y[i + 1].abs() < y[i].abs()
            //     {
            //         roots.push(i + 1);
            //     }
            //     else
            //     {
            //         roots.push(i);
            //     }
            // }
            let dyp = y[i + 1] - y[i];
            let dym = y[i] - y[i - 1];
            let convex = dyp < dym;
            let concave = dyp > dym;
            //check for extremes
            if y[i + 1] < y[i] && y[i - 1] < y[i]
            {
                extremals.push(i);
                if glob_max < y[i].abs()
                {
                    glob_max = y[i].abs();
                }
            }
            else if y[i + 1] > y[i] && y[i - 1] > y[i]
            {
                extremals.push(i);
                if glob_max < y[i].abs()
                {
                    glob_max = y[i].abs();
                }
            }
            else if (concave && prev_convex) || (convex && prev_concave)
            {
                //check for inflections
                extremals.push(i)
            }

            prev_concave = concave;
            prev_convex = convex;
        }
        let mut extr_iter = (0..extremals.len()).into_iter();
        let threshold = glob_max * 0.1;
        extr_iter.try_fold(0, |_acc, i| {
            if y[extremals[i]].abs() > threshold
            {
                None
            }
            else
            {
                Some(0)
            }
        });
        let start = extr_iter.start - 1;
        extr_iter.try_rfold(0, |_acc, i| {
            if y[extremals[i]].abs() > threshold
            {
                None
            }
            else
            {
                Some(0)
            }
        });

        let stop = extr_iter.end + 1;
        let extremals = &extremals[start..stop];
        let (x, y) : (Vec<f64>, Vec<f64>) =
            extremals.into_iter().map(|&i| (x[i] as f64, y[i])).unzip();
        //interpolate wavelet function using extremal points
        let cubic_spline = CubicSpline::interpolate(&x, &y, CubicSplineConstraint::Natural);
        let cubic_spline_d1 = cubic_spline.differentiate();
        let cubic_spline_d2 = cubic_spline_d1.differentiate();
        let cubic_spline_d3 = cubic_spline_d2.differentiate();
        //`cubic_spline_d3` is picewise constant, with discontinuities at `x`
        //mathru implementation of cubic spline at this points evaluates to right-side value (x => x+)
        //  (exept last x which evals to left-side val (x_last=> x_last-))
        let (x0f, xJf) = (*x.first().unwrap(), *x.last().unwrap());
        let (x0, xJ) = (x0f as isize, xJf as isize);
        let mut d3psi_raw : Vec<Point<f64>> = x
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
        let init_calc_half_len = scale * 60;
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
        if self.wavelets.len() < scale
        {
            self.wavelets.resize(scale, None);
        }
        if let Some(poly_wise) = self.wavelets[scale - 1].clone()
        {
            return poly_wise;
        }
        else
        {
            let poly_wise = PolyWiseComplex::new(self.func, scale, 0.1); //TODO check time step
            self.wavelets[scale - 1] = Some(poly_wise.clone());
            return poly_wise;
        }
    }
}

#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Debug,
    Default,
    strum_macros::AsRefStr,
    EnumString,
    EnumVariantNames,
)]
#[strum(serialize_all = "title_case")]
pub enum WaveletType
{
    #[default]
    Morlet,
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
        HashMap::from([(
            Morlet,
            WaveletParams {
                func :     &morlet_wavelet_func,
                wavelets : Vec::new(),
            },
        )])
    }
}

fn morlet_wavelet_func(t : &f64, s : &f64) -> (f64, f64)
{
    let fb = s * 0.25; //=sigma^2 /8
    let fc = (2.0 / 2_f64.ln()).sqrt() / s;
    let ret = Complex64::new(
        (1.0 / (PI * fb).sqrt()) * (-1.0 * (t).powi(2) / fb).exp(),
        0.0,
    ) * Complex64::new(0.0, 2.0 * PI * t * fc).exp();
    (ret.re, ret.im)
}
