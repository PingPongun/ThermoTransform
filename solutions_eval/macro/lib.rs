extern crate proc_macro;

use proc_macro::TokenStream;
#[proc_macro]
pub fn antiderivative_report_delta(item : TokenStream) -> TokenStream
{
    let name = item.to_string();
    "
    let mut antiderivative_### = vec![];
    c.bench_function(\"###\", |b| {
        b.iter(|| antiderivative_###=###(black_box(&derivative)))
    });
        let deltas_###_iter = antiderivative_###
    .iter()
    .zip(antiderivative.iter())
    .map(|(calc, refr)| refr - calc);
let relative_deltas_###_iter = deltas_###_iter
.clone()
.zip(antiderivative.iter())
.map(|(delta, refr)| delta.abs()/(refr.abs()+1.));
println!(
    \"### max delta:{}\",
    deltas_###_iter
        .clone()
        .reduce(|x, y| f64::max(x, y.abs()))
        .unwrap()
);
println!(
    \"### sum delta:{}\",
    deltas_###_iter.clone().reduce(|x, y| x + y).unwrap()
);
println!(
    \"### sum abs delta:{}\",
    deltas_###_iter.reduce(|x, y| x + y.abs()).unwrap()
);
println!(
    \"### max relative delta:{}\",
    relative_deltas_###_iter
        .clone()
        .reduce(|x, y| f64::max(x, y.abs()))
        .unwrap()
    );
    println!(
        \"### sum relative delta:{}\",
        relative_deltas_###_iter.clone().reduce(|x, y| x + y).unwrap()
    );"
    .replace("###", &name)
    .parse()
    .unwrap()
}
#[proc_macro]
pub fn antiderivative_report_delta_f32(item : TokenStream) -> TokenStream
{
    let name = item.to_string();
    "
    let derivative:Vec<f32>=derivative.iter().cloned().map(|x| x as f32).collect();
    let antiderivative:Vec<f32>=antiderivative.iter().cloned().map(|x| x as f32).collect();
    let mut antiderivative_### = vec![];
    c.bench_function(\"###\", |b| {
        b.iter(|| antiderivative_###=###(black_box(&derivative)))
    });
        let deltas_###_iter = antiderivative_###
    .iter()
    .zip(antiderivative.iter())
    .map(|(calc, refr)| refr - calc);
let relative_deltas_###_iter = deltas_###_iter
.clone()
.zip(antiderivative.iter())
.map(|(delta, refr)| delta.abs()/(refr.abs()+1.));
println!(
    \"### max delta:{}\",
    deltas_###_iter
        .clone()
        .reduce(|x, y| f32::max(x, y.abs()))
        .unwrap()
);
println!(
    \"### sum delta:{}\",
    deltas_###_iter.clone().reduce(|x, y| x + y).unwrap()
);
println!(
    \"### sum abs delta:{}\",
    deltas_###_iter.reduce(|x, y| x + y.abs()).unwrap()
);
println!(
    \"### max relative delta:{}\",
    relative_deltas_###_iter
        .clone()
        .reduce(|x, y| f32::max(x, y.abs()))
        .unwrap()
    );
    println!(
        \"### sum relative delta:{}\",
        relative_deltas_###_iter.clone().reduce(|x, y| x + y).unwrap()
    );"
    .replace("###", &name)
    .parse()
    .unwrap()
}
/*
x,x+r,x+2r,x+3r,x+4r
ax^4+bx^3+cx^2+dx+e
->integral-> 0.25ax^5+0.33bx^4+0.5cx^3+dx^2+ex+const
y1=ax^4+bx^3+cx^2+dx+e
y2=a(x+r)^4+b(x+r)^3+c(x+r)^2+d(x+r)+e




x,x+r,x+2r
cx^2+dx+e
->integral-> 0.5cx^3+dx^2+ex+const
y1=cx^2+dx+e
y2=c(x+r)^2+d(x+r)+e
y3=c(x+2r)^2+d(x+2r)+e
*/
