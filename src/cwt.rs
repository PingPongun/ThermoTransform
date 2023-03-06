use ndarray::Array2;
use ndarray::Array3;
use ndarray::ArrayView1;
use ndarray::ArrayViewMut1;
use ndarray::Axis;
use rayon::prelude::IndexedParallelIterator;
use rayon::prelude::IntoParallelIterator;
use rayon::prelude::ParallelIterator;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use strum_macros::{EnumString, EnumVariantNames};

use crate::tt_common::*;
use crate::wavelet::*;

#[derive(
    Clone, Copy, PartialEq, Debug, Default, strum_macros::AsRefStr, EnumString, EnumVariantNames,
)]
#[strum(serialize_all = "title_case")]
pub enum WtResultMode
{
    #[default]
    Phase,
    Magnitude,
    Real,
    Imaginary,
}
type Integrals = [f64; 4];

fn gen_integrals(non_integrated : ArrayView1<'_, f64>, mut integrals : ArrayViewMut1<'_, Integrals>)
{
    // // OPTION 1: Bool's integration
    // const DIV1 : f64 = 1. / 2. / 10.;
    // const DIV2 : f64 = 1. / 3. / 10.;
    // const DIV3 : f64 = 3. / 8. / 10.;
    // const DIV7 : f64 = 7. * 2. / 45. / 10.;
    // const DIV12 : f64 = 12. * 2. / 45. / 10.;
    // const DIV32 : f64 = 32. * 2. / 45. / 10.;
    // //first integration
    // integrals[0][0] = 0.0;
    // integrals[1][0] = DIV1 * (non_integrated[0] + non_integrated[1]);
    // integrals[2][0] = DIV2 * (non_integrated[0] + 4. * non_integrated[1] + non_integrated[2]);
    // integrals[3][0] = DIV3
    //     * (non_integrated[0] + 3. * non_integrated[1] + 3. * non_integrated[2] + non_integrated[3]);
    // for t in 4..non_integrated.len()
    // {
    //     integrals[t][0] = (DIV7 * non_integrated[t]
    //         + DIV32 * non_integrated[t - 1]
    //         + DIV12 * non_integrated[t - 2]
    //         + DIV32 * non_integrated[t - 3]
    //         + DIV7 * non_integrated[t - 4])
    //         + integrals[t - 4][0]
    // }
    // //further integrations
    // for integral in 0..3
    // {
    //     integrals[0][integral + 1] = 0.0;
    //     integrals[1][integral + 1] = DIV1 * (integrals[0][integral] + integrals[1][integral]);
    //     integrals[2][integral + 1] =
    //         DIV2 * (integrals[0][integral] + 4. * integrals[1][integral] + integrals[2][integral]);
    //     integrals[3][integral + 1] = DIV3
    //         * (integrals[0][integral]
    //             + 3. * integrals[1][integral]
    //             + 3. * integrals[2][integral]
    //             + integrals[3][integral]);
    //     for t in 4..non_integrated.len()
    //     {
    //         integrals[t][integral + 1] = (DIV7 * integrals[t][integral]
    //             + DIV32 * integrals[t - 1][integral]
    //             + DIV12 * integrals[t - 2][integral]
    //             + DIV32 * integrals[t - 3][integral]
    //             + DIV7 * integrals[t - 4][integral])
    //             + integrals[t - 4][integral + 1]
    //     }
    // }

    // // OPTION 2: trapezoid integration
    const DIV1 : f64 = 1. / 2. / 10.;
    //first integration
    integrals[0][0] = 0.0;
    for t in 1..non_integrated.len()
    {
        integrals[t][0] = DIV1 * (non_integrated[t - 1] + non_integrated[t]) + integrals[t - 1][0]
    }
    //further integrations
    for integral in 0..3
    {
        integrals[0][integral + 1] = 0.0;
        for t in 1..non_integrated.len()
        {
            integrals[t][integral + 1] = DIV1
                * (integrals[t - 1][integral] + integrals[t][integral])
                + integrals[t - 1][integral + 1]
        }
    }
}

pub struct TTLazyCWT
{
    integrals : Array3<Integrals>,
}
impl TTLazyCWT
{
    pub fn new(input : &Array3<f64>, file_state : Arc<AtomicFileState>) -> Option<TTLazyCWT>
    {
        let shape_raw = input.raw_dim();
        let mut integrated = TTLazyCWT {
            integrals : Array3::<Integrals>::default(shape_raw),
        };

        let integrated_iter = integrated
            .integrals
            .lanes_mut(Axis(0))
            .into_iter()
            .into_par_iter();
        let time_iter = input.lanes(Axis(0)).into_iter().into_par_iter();
        let iter = IndexedParallelIterator::zip(time_iter, integrated_iter);
        iter.for_each(|(lane_non_integrated, lane_integrals)| {
            if FileState::Processing == file_state.load(Ordering::Relaxed)
            {
                gen_integrals(lane_non_integrated, lane_integrals);
            }
        });

        if FileState::Processing == file_state.load(Ordering::Relaxed)
        {
            Some(integrated)
        }
        else
        {
            None
        }
    }
    pub fn cwt(&self, wavelet_bank : &mut WaveletBank, params : &TransformViewParams)
        -> Array2<f64>
    {
        Array2::default((self.integrals.dim().1, self.integrals.dim().2))
    }
}
