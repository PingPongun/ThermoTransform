use ndarray::s;
use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut3;
use ndarray::Zip;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use rayon::slice::ParallelSliceMut;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::tt_common::*;
use crate::tt_fourier::*;
use crate::wavelet::*;

pub struct TTLazyCWT
{
    integrals : [Array3<f64>; 4],
}
impl TTLazyCWT
{
    pub fn new(fourier : &TTFourier, file_state : Arc<AtomicFileState>) -> Option<TTLazyCWT>
    {
        let integrated = TTLazyCWT {
            integrals : fourier.integrals(),
        };

        if FileState::ProcessingWavelet == file_state.load(Ordering::Relaxed)
        {
            Some(integrated)
        }
        else
        {
            None
        }
    }
    pub fn cwt(
        &self,
        wavelet_bank : &mut WaveletBank,
        params : &ViewMode,
        settings : &GlobalSettings,
    ) -> Array2<f64>
    {
        let wavelet = wavelet_bank
            .get_mut(&params.wavelet.load(Ordering::Relaxed))
            .unwrap(); //`WaveletBank` should have entries for all WaveletType-enum wariants, if not this is an error in code so panic is apropriate
        let XYTshape = self.integrals[0].dim();
        let cwt_fn = |integrals : (
            ArrayView1<'_, f64>,
            ArrayView1<'_, f64>,
            ArrayView1<'_, f64>,
            ArrayView1<'_, f64>,
        ),
                      polywise : &PolyWiseComplex,
                      t : isize| {
            let real_img : Vec<f64> = polywise
                .0
                .iter()
                .map(|polywise| {
                    let mut accum : f64 = 0.0;
                    let s0 = t + polywise.dxpsi[0].x;
                    let sJ = t + polywise.dxpsi[1].x;
                    let integral_s0;
                    let integral_sJ;
                    let last_idx = integrals.0.len() - 1;
                    let last_idx_isize = last_idx as isize;
                    if s0 < 0
                    {
                        //wavelet "goes beyond" signal => extend signal using antisymetric half-point extension
                        //x0, x1, x2 -> (2*x0-x2), (2*x0-x1), x0, x1, x2
                        let s0 = -s0;
                        if s0 > last_idx_isize
                        {
                            integral_s0 = [0., 0., 0.];
                        }
                        else
                        {
                            let s0 = s0 as usize;
                            integral_s0 = [
                                f64::mul_add(integrals.0[0], 2.0, -integrals.0[s0]),
                                f64::mul_add(integrals.1[0], 2.0, -integrals.1[s0]),
                                f64::mul_add(integrals.2[0], 2.0, -integrals.2[s0]),
                            ];
                        }
                    }
                    else
                    {
                        let s0 = s0 as usize;
                        integral_s0 = [integrals.0[s0], integrals.1[s0], integrals.2[s0]];
                    }
                    if sJ > last_idx_isize
                    {
                        //wavelet "goes beyond" signal ...
                        let sJ = 2 * last_idx_isize - sJ;
                        if sJ < 0
                        {
                            integral_sJ = [0., 0., 0.];
                        }
                        else
                        {
                            let sJ = sJ as usize;
                            integral_sJ = [
                                f64::mul_add(integrals.0[last_idx], 2.0, -integrals.0[sJ]),
                                f64::mul_add(integrals.1[last_idx], 2.0, -integrals.1[sJ]),
                                f64::mul_add(integrals.2[last_idx], 2.0, -integrals.2[sJ]),
                            ];
                        }
                    }
                    else
                    {
                        let sJ = sJ as usize;
                        integral_sJ = [integrals.0[sJ], integrals.1[sJ], integrals.2[sJ]];
                    }
                    accum = f64::mul_add(integral_s0[0], polywise.dxpsi[0].y[0], accum);
                    accum = f64::mul_add(integral_s0[1], polywise.dxpsi[0].y[1], accum);
                    accum = f64::mul_add(integral_s0[2], polywise.dxpsi[0].y[2], accum);
                    accum = f64::mul_add(integral_sJ[0], polywise.dxpsi[1].y[0], accum);
                    accum = f64::mul_add(integral_sJ[1], polywise.dxpsi[1].y[1], accum);
                    accum = f64::mul_add(integral_sJ[2], polywise.dxpsi[1].y[2], accum);
                    polywise.d3psi.iter().for_each(|point| {
                        let s = t + point.x;
                        if s <= last_idx_isize && s >= 0
                        {
                            accum = f64::mul_add(integrals.3[s as usize], point.y, accum)
                        }
                        else if s < 0 && s >= -last_idx_isize
                        {
                            accum = f64::mul_add(
                                f64::mul_add(integrals.3[0], 2.0, -integrals.3[(-s) as usize]),
                                point.y,
                                accum,
                            )
                        }
                        else if s <= last_idx_isize * 2 && s > last_idx_isize
                        {
                            accum = f64::mul_add(
                                f64::mul_add(
                                    integrals.3[last_idx],
                                    2.0,
                                    -integrals.3[(2 * last_idx_isize - s) as usize],
                                ),
                                point.y,
                                accum,
                            )
                        }
                        else
                        {
                            ()
                        }
                    });
                    accum
                })
                .collect();
            //convert to requested format
            match params.display_mode.load(Ordering::Relaxed)
            {
                ComplexResultMode::Phase => real_img[1].atan2(real_img[0]), //radians
                ComplexResultMode::Magnitude => real_img[0].hypot(real_img[1]),
                ComplexResultMode::Real => real_img[0],
                ComplexResultMode::Imaginary => real_img[1],
            }
        };
        let view_axes = params.get_view_axes();
        let mut size = [0, 0];
        let position = params.position.read().clone();
        let roi_zoom = settings.roi_zoom.load(Ordering::Relaxed);
        let x_range = if view_axes[0] == TTAxis::X
        {
            let range = if roi_zoom
            {
                settings.get_roi(TTAxis::X)
            }
            else
            {
                0..XYTshape.0
            };
            size[0] = range.len();
            range
        }
        else
        {
            position[TTAxis::X as usize]..position[TTAxis::X as usize] + 1
        };
        let y_range = if view_axes[1] == TTAxis::Y
        {
            let range = if roi_zoom
            {
                settings.get_roi(TTAxis::Y)
            }
            else
            {
                0..XYTshape.1
            };
            size[1] = range.len();
            range
        }
        else
        {
            position[TTAxis::Y as usize]..position[TTAxis::Y as usize] + 1
        };
        let t_chunk_div;
        let t_range = if view_axes[0] == TTAxis::T || view_axes[1] == TTAxis::T
        {
            let range = if roi_zoom
            {
                settings.get_roi(TTAxis::T)
            }
            else
            {
                0..XYTshape.2
            };
            size[(view_axes[1] == TTAxis::T) as usize] = range.len();
            t_chunk_div = range.len();
            range
        }
        else
        {
            t_chunk_div = 1;
            position[TTAxis::T as usize]..position[TTAxis::T as usize] + 1
        };
        let s_chunk_div;
        let s_range = if view_axes[0] == TTAxis::S || view_axes[1] == TTAxis::S
        {
            let range = if roi_zoom
            {
                settings.get_roi(TTAxis::S)
            }
            else
            {
                0..XYTshape.2
            };
            wavelet.batch_calc(range.end);
            size[(view_axes[1] == TTAxis::S) as usize] = range.len();
            s_chunk_div = range.len();
            range
        }
        else
        {
            s_chunk_div = 1;
            let _ = wavelet.get_poly_wise(position[TTAxis::S as usize]);
            position[TTAxis::S as usize]..position[TTAxis::S as usize] + 1
        };
        let avm3 = (x_range.len(), y_range.len(), 1);
        let slice_arg = s![x_range, y_range, ..];
        let mut v : Vec<f64> = vec![0.0; size[0] * size[1]];
        s_range
            .into_par_iter()
            .zip(v.par_chunks_exact_mut(size[0] * size[1] / s_chunk_div))
            .for_each(|(s, v)| {
                let polywise = wavelet.uget_poly_wise(s as usize);
                t_range
                    .clone()
                    .into_par_iter()
                    .zip(v.par_chunks_exact_mut(v.len() / t_chunk_div))
                    .for_each(|(t, v)| {
                        Zip::from(self.integrals[0].slice(slice_arg).lanes(AXIS_T))
                            .and(self.integrals[1].slice(slice_arg).lanes(AXIS_T))
                            .and(self.integrals[2].slice(slice_arg).lanes(AXIS_T))
                            .and(self.integrals[3].slice(slice_arg).lanes(AXIS_T))
                            .and(
                                ArrayViewMut3::from_shape(avm3, v)
                                    .unwrap()
                                    .lanes_mut(AXIS_T),
                            )
                            .par_for_each(|i0, i1, i2, i3, mut v| {
                                v[0] = cwt_fn((i0, i1, i2, i3), &polywise, t as isize)
                            })
                    })
            });
        //if X-t, X-s, t-s transpose
        match view_axes
        {
            [TTAxis::X, TTAxis::T] | [TTAxis::X, TTAxis::S] | [TTAxis::T, TTAxis::S] =>
            {
                Array2::from_shape_vec((size[1], size[0]), v)
                    .unwrap()
                    .reversed_axes()
            }
            [TTAxis::X, TTAxis::Y] => Array2::from_shape_vec((size[0], size[1]), v).unwrap(),
            [TTAxis::T, TTAxis::Y] | [TTAxis::S, TTAxis::Y] =>
            {
                Array2::from_shape_vec((size[0], size[1]), v)
                    .unwrap()
                    .reversed_axes() //counter transposition in TTViewBackend::update_image()
            }
            _ => unreachable!(),
        }
    }
}

impl TTInputData
{
    pub fn cwt(
        &self,
        wavelet_bank : &mut WaveletBank,
        params : &ViewMode,
        settings : &GlobalSettings,
    ) -> Array2<f64>
    {
        let wavelet = wavelet_bank
            .get_mut(&params.wavelet.load(Ordering::Relaxed))
            .unwrap(); //`WaveletBank` should have entries for all WaveletType-enum wariants, if not this is an error in code so panic is apropriate
        let view_axes = params.get_view_axes();
        let mut size = [0, 0];
        let position = params.position.read().clone();
        let roi_zoom = settings.roi_zoom.load(Ordering::Relaxed);
        let x_range = if view_axes[0] == TTAxis::X
        {
            let range = if roi_zoom
            {
                settings.get_roi(TTAxis::X)
            }
            else
            {
                0..self.width
            };
            size[0] = range.len();
            range
        }
        else
        {
            position[TTAxis::X as usize]..position[TTAxis::X as usize] + 1
        };
        let y_range = if view_axes[1] == TTAxis::Y
        {
            let range = if roi_zoom
            {
                settings.get_roi(TTAxis::Y)
            }
            else
            {
                0..self.height
            };
            size[1] = range.len();
            range
        }
        else
        {
            position[TTAxis::Y as usize]..position[TTAxis::Y as usize] + 1
        };
        let t_chunk_div;
        let t_range = if view_axes[0] == TTAxis::T || view_axes[1] == TTAxis::T
        {
            let range = if roi_zoom
            {
                settings.get_roi(TTAxis::T)
            }
            else
            {
                0..self.frames
            };
            size[(view_axes[1] == TTAxis::T) as usize] = range.len();
            t_chunk_div = range.len();
            range
        }
        else
        {
            t_chunk_div = 1;
            position[TTAxis::T as usize]..position[TTAxis::T as usize] + 1
        };
        let s_chunk_div;
        let s_range = if view_axes[0] == TTAxis::S || view_axes[1] == TTAxis::S
        {
            let range = if roi_zoom
            {
                settings.get_roi(TTAxis::S)
            }
            else
            {
                0..self.frames
            };
            wavelet.batch_calc(range.end);
            size[(view_axes[1] == TTAxis::S) as usize] = range.len();
            s_chunk_div = range.len();
            range
        }
        else
        {
            s_chunk_div = 1;
            let _ = wavelet.get_poly_wise(position[TTAxis::S as usize]);
            position[TTAxis::S as usize]..position[TTAxis::S as usize] + 1
        };
        let avm3 = (x_range.len(), y_range.len(), 1);
        let slice_arg = s![x_range, y_range, ..];
        let mut v : Vec<f64> = vec![0.0; size[0] * size[1]];
        s_range
            .into_par_iter()
            .zip(v.par_chunks_exact_mut(size[0] * size[1] / s_chunk_div))
            .for_each(|(s, v)| {
                let polywise = wavelet.uget_poly_wise(s as usize);
                t_range
                    .clone()
                    .into_par_iter()
                    .zip(v.par_chunks_exact_mut(v.len() / t_chunk_div))
                    .for_each(|(t, v)| {
                        Zip::from(self.data.slice(slice_arg).lanes(AXIS_T))
                            .and(
                                ArrayViewMut3::from_shape(avm3, v)
                                    .unwrap()
                                    .lanes_mut(AXIS_T),
                            )
                            .par_for_each(|data, mut v| {
                                let real_img : Vec<f64> = polywise
                                    .0
                                    .iter()
                                    .map(|polywise| {
                                        let mid = polywise.psi.len() >> 1;
                                        let t = t as usize;
                                        let mut accum : f64 = 0.0;

                                        let (minp, mins) = if t < mid
                                        {
                                            //extend below 0
                                            let minp = mid - t;
                                            let mins = minp.clamp(0, self.frames - 1);
                                            (1..=mins).rev().into_iter().enumerate().for_each(
                                                |(ip, is)| {
                                                    let psi = polywise.psi[ip];
                                                    let sig = f64::mul_add(data[0], 2.0, -data[is]);
                                                    accum = f64::mul_add(psi, sig, accum);
                                                },
                                            );
                                            (minp, 0)
                                        }
                                        else
                                        {
                                            (0, t - mid)
                                        };

                                        let (maxp, maxs) = if t + mid >= self.frames
                                        {
                                            //extend above max
                                            let a = t + mid - self.frames;
                                            let maxp = polywise.psi.len() - a;
                                            let maxs = polywise.psi.len().saturating_sub(a);
                                            (maxp..polywise.psi.len())
                                                .into_iter()
                                                .zip((maxs..self.frames).rev())
                                                .for_each(|(ip, is)| {
                                                    let psi = polywise.psi[ip];
                                                    let sig = f64::mul_add(
                                                        data[self.frames - 1],
                                                        2.0,
                                                        -data[is],
                                                    );
                                                    accum = f64::mul_add(psi, sig, accum);
                                                });
                                            (maxp, self.frames)
                                        }
                                        else
                                        {
                                            (polywise.psi.len(), t + mid)
                                        };
                                        (minp..maxp).into_iter().zip(mins..maxs).for_each(
                                            |(ip, is)| {
                                                let psi = polywise.psi[ip];
                                                let sig = data[is];
                                                accum = f64::mul_add(psi, sig, accum);
                                            },
                                        );
                                        accum
                                    })
                                    .collect();
                                //convert to requested format
                                v[0] = match params.display_mode.load(Ordering::Relaxed)
                                {
                                    ComplexResultMode::Phase => real_img[1].atan2(real_img[0]), //radians
                                    ComplexResultMode::Magnitude => real_img[0].hypot(real_img[1]),
                                    ComplexResultMode::Real => real_img[0],
                                    ComplexResultMode::Imaginary => real_img[1],
                                }
                            })
                    })
            });
        //if X-t, X-s, t-s transpose
        match view_axes
        {
            [TTAxis::X, TTAxis::T] | [TTAxis::X, TTAxis::S] | [TTAxis::T, TTAxis::S] =>
            {
                Array2::from_shape_vec((size[1], size[0]), v)
                    .unwrap()
                    .reversed_axes()
            }
            [TTAxis::X, TTAxis::Y] => Array2::from_shape_vec((size[0], size[1]), v).unwrap(),
            [TTAxis::T, TTAxis::Y] | [TTAxis::S, TTAxis::Y] =>
            {
                Array2::from_shape_vec((size[0], size[1]), v)
                    .unwrap()
                    .reversed_axes() //counter transposition in TTViewBackend::update_image()
            }
            _ => unreachable!(),
        }
    }
}
