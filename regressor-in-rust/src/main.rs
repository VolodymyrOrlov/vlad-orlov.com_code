use nalgebra::{DMatrix, Scalar};

use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::str::FromStr;

use smartcore::model_selection::train_test_split;
use smartcore::metrics::mean_absolute_error;

fn parse_csv<N, R>(input: R) -> Result<DMatrix<N>, Box<dyn std::error::Error>>
  where N: FromStr + Scalar,
        N::Err: std::error::Error,
        R: BufRead
{
  // initialize an empty vector to fill with numbers
  let mut data = Vec::new();

  // initialize the number of rows to zero; we'll increment this
  // every time we encounter a newline in the input
  let mut rows = 0;

  // for each line in the input,
  for line in input.lines() {
    // increment the number of rows
    rows += 1;
    // iterate over the items in the row, separated by commas
    for datum in line?.split_terminator(",") {
      // trim the whitespace from the item, parse it, and push it to
      // the data array
      data.push(N::from_str(datum.trim())?);
    }
  }

  // The number of items divided by the number of rows equals the
  // number of columns.
  let cols = data.len() / rows;

  // Construct a `DMatrix` from the data in the vector.
  Ok(DMatrix::from_row_slice(rows, cols, &data[..]))
}

fn main() { 
    let file = File::open("boston.csv").unwrap();
    let bos: DMatrix<f64> = parse_csv(BufReader::new(file)).unwrap();
    println!("{}", bos.rows(0, 5));

    let x = bos.columns(0, 13).into_owned();
    let y = bos.column(13).into_owned();
    
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y.transpose(), 0.2);

    let a = x_train.clone().insert_column(13, 1.0).into_owned();
    let b = y_train.clone().transpose();
            
    // np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    let x = (a.transpose() * &a).try_inverse().unwrap() * &a.transpose() * &b;
    let coeff = x.rows(0, 13);
    let intercept = x[(13, 0)];

    println!("coeff: {}, intercept: {}", coeff, intercept);

    // Q, R = np.linalg.qr(A)
    let qr = a.qr();
    let (q, r) = (qr.q().transpose(), qr.r());        
    let x = r.try_inverse().unwrap() * &q * &b;
    let coeff = x.rows(0, 13);
    let intercept = x[(13, 0)];

    println!("coeff: {}, intercept: {}", coeff, intercept);

    let y_hat = (x_test * &coeff).add_scalar(intercept);

    println!("mae: {}", mean_absolute_error(&y_test, &y_hat.transpose()));  
}
