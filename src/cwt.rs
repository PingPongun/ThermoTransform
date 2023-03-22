use ndarray::Array2;
use ndarray::Array3;
use ndarray::Axis;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
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
    pub fn cwt(&self, wavelet_bank : &mut WaveletBank, params : &WaveletViewParams) -> Array2<f64>
    {
        let wavelet = wavelet_bank.get_mut(&params.wavelet).unwrap(); //`WaveletBank` should have entries for all WaveletType-enum wariants, if not this is an error in code so panic is apropriate
        let polywise = wavelet.get_poly_wise(params.scale.val);
        let shape = self.integrals[0].dim();
        let t = params.time.val as isize;
        let v = self.integrals[0]
            .lanes(Axis(0))
            .into_iter()
            .into_par_iter()
            .zip(self.integrals[1].lanes(Axis(0)).into_iter())
            .zip(self.integrals[2].lanes(Axis(0)).into_iter())
            .zip(self.integrals[3].lanes(Axis(0)).into_iter())
            .map(|integrals| {
                let integrals = (
                    integrals.0 .0 .0,
                    integrals.0 .0 .1,
                    integrals.0 .1,
                    integrals.1,
                );
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
                            let s0 = (-s0) as usize;
                            integral_s0 = [
                                f64::mul_add(integrals.0[0], 2.0, -integrals.0[s0]),
                                f64::mul_add(integrals.1[0], 2.0, -integrals.1[s0]),
                                f64::mul_add(integrals.2[0], 2.0, -integrals.2[s0]),
                            ];
                        }
                        else
                        {
                            let s0 = s0 as usize;
                            integral_s0 = [integrals.0[s0], integrals.1[s0], integrals.2[s0]];
                        }
                        if sJ > last_idx_isize
                        {
                            //wavelet "goes beyond" signal ...
                            let sJ = (2 * last_idx_isize - sJ) as usize;
                            integral_sJ = [
                                f64::mul_add(integrals.0[last_idx], 2.0, -integrals.0[sJ]),
                                f64::mul_add(integrals.1[last_idx], 2.0, -integrals.1[sJ]),
                                f64::mul_add(integrals.2[last_idx], 2.0, -integrals.2[sJ]),
                            ];
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
                            if s < 0
                            {
                                accum = f64::mul_add(
                                    f64::mul_add(integrals.3[0], 2.0, -integrals.3[(-s) as usize]),
                                    point.y,
                                    accum,
                                )
                            }
                            else if s > last_idx_isize
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
                                accum = f64::mul_add(integrals.3[s as usize], point.y, accum)
                            }
                        });
                        accum
                    })
                    .collect();
                //convert to requested format
                match params.display_mode
                {
                    ComplexResultMode::Phase => real_img[1].atan2(real_img[0]), //radians
                    ComplexResultMode::Magnitude => real_img[0].hypot(real_img[1]),
                    ComplexResultMode::Real => real_img[0],
                    ComplexResultMode::Imaginary => real_img[1],
                }
            })
            .collect();
        Array2::from_shape_vec((shape.1, shape.2), v).unwrap()
    }
}
