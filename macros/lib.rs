extern crate proc_macro;
use std::fmt::Write;

#[proc_macro]
pub fn array_pows_2_3(_input : proc_macro::TokenStream) -> proc_macro::TokenStream
{
    let mut out = String::new();
    let max2 = usize::MAX.checked_ilog(2).unwrap();
    let max3 = usize::MAX.checked_ilog(3).unwrap();
    let mut primes : Vec<_> = (0..=max3)
        .into_iter()
        .map(|x3| {
            let pow3 = 3_usize.pow(x3);
            let x2max = max2 - x3;
            (0..=x2max)
                .into_iter()
                .map(move |x2| usize::checked_mul(2_usize.pow(x2), pow3))
                .filter(|x| x.is_some())
                .map(|x| x.unwrap())
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect();
    primes.sort_unstable();
    primes.dedup();
    let _ = write!(&mut out, "&[");
    primes.iter().for_each(|x| {
        let _ = write!(&mut out, "{},", x);
    });
    let _ = write!(&mut out, "]");

    out.parse().unwrap()
}
