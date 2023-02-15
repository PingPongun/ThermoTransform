extern crate criterion;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
extern crate solutions_eval_macros;
use solutions_eval_macros::*;
use std::f64::consts::E;
use std::iter;

fn integrate(input : &[f64]) -> Vec<f64>
{
    //~83us
    //2.04561E+28 max delta
    //-2.14018E+30 sum delta
    //2.14018E+30 sum abs delta

    let mut acumulator = 0.0;
    let temp = iter::once(0.0).chain(input.windows(2).map(|x| {
        acumulator += (x[0] + x[1]) / 2. / 10.;
        acumulator
    }));
    temp.collect()
}
fn integrate2(input : &[f64]) -> Vec<f64>
{
    //~175us
    // 3.05524E+19 max delta
    // -5.07211E+20 sum delta
    // 1.23904E+21 sum abs delta

    let mut acumulator = [0.0; 5];
    let binding = [
        0.0,
        (input[0] + input[1]) / 2. / 10.,
        (input[0] + 4. * input[1] + input[2]) / 3. / 10.,
        (input[0] + 3. * input[1] + 3. * input[2] + input[3]) * 3. / 8. / 10.,
    ];
    acumulator = [binding[0], binding[1], binding[2], binding[3], 0.];
    let firsts = binding.iter().cloned();
    let temp = firsts.chain(input.windows(5).map(|x| {
        acumulator[4] =
            (7. * x[0] + 32. * x[1] + 12. * x[2] + 32. * x[3] + 7. * x[4]) * 2. / 45. / 10.
                + acumulator[0];
        acumulator[0] = acumulator[1];
        acumulator[1] = acumulator[2];
        acumulator[2] = acumulator[3];
        acumulator[3] = acumulator[4];
        acumulator[4]
    }));
    temp.collect()
}
fn integrate3(input : &[f64]) -> Vec<f64>
{
    //~50us

    let mut out = vec![0.0; input.len()];
    out[0] = 0.0;
    const div1 : f64 = 1. / 2. / 10.;
    out[1] = (div1 * (input[0] + input[1]));
    const div2 : f64 = 1. / 3. / 10.;
    out[2] = (div2 * (input[0] + 4. * input[1] + input[2]));
    const div3 : f64 = 3. / 8. / 10.;
    out[3] = (div3 * (input[0] + 3. * input[1] + 3. * input[2] + input[3]));
    const div7 : f64 = 7. * 2. / 45. / 10.;
    const div12 : f64 = 12. * 2. / 45. / 10.;
    const div32 : f64 = 32. * 2. / 45. / 10.;
    for i in 4..input.len()
    {
        out[i] = ((div7 * input[i]
            + div32 * input[i - 1]
            + div12 * input[i - 2]
            + div32 * input[i - 3]
            + div7 * input[i - 4])
            + out[i - 4])
    }
    out
}

const LEN : usize = 10003;
fn derivative_generator(x : f64) -> f64
{
    (10. * (E.powf(x / 10.) * (x.powf(2.) - 40. * x + 1.)
        - 10. * (x.powf(2.) - 3.) * (x.powf(2.)) * f64::sin(x)
        + 10. * (x.powf(5.) + x.powf(3.)) * f64::cos(x)))
        / (x.powf(2.) + 1.).powf(3.)
}
fn antiderivative_generator(x : f64) -> f64
{
    (x.powf(3.) * f64::sin(x) + E.powf(x / 10.)) * (10. / (x.powf(2.) + 1.)).powf(2.)
}
// fn derivative_generator(x : f64) -> f64 { f64::cos(x) }
// fn antiderivative_generator(x : f64) -> f64 { f64::sin(x) }
fn criterion_benchmark(c : &mut Criterion)
{
    let derivative : [f64; LEN] =
        array_init::array_init(|x| derivative_generator(x as f64 / 10_f64));
    let antiderivative : [f64; LEN] =
        array_init::array_init(|x| antiderivative_generator(x as f64 / 10_f64));
    println!("{}    {}", derivative[LEN - 1], antiderivative[LEN - 1]); //2.56E+32    2.66243E+33
    antiderivative_report_delta!(integrate);
    antiderivative_report_delta!(integrate2);
    antiderivative_report_delta!(integrate3);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
