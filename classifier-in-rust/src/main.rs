use argmin::prelude::*;
use argmin::solver::linesearch::{ArmijoCondition, BacktrackingLineSearch};
use argmin::solver::quasinewton::LBFGS;
use nalgebra::{DMatrix, DVector, Scalar};

use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::str::FromStr;

use smartcore::metrics::roc_auc_score;
use smartcore::model_selection::train_test_split;

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

struct BinaryObjectiveFunction<'a> {
    x: &'a DMatrix<f64>,
    y: &'a DVector<f64>
}

impl<'a> ArgminOp for BinaryObjectiveFunction<'a> {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = Vec<Vec<f64>>;
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, w: &Self::Param) -> Result<Self::Output, Error> {
        let mut f = 0f64;
        let (n, _) = self.x.shape();

        for i in 0..n {
            let wx = dot(w, &self.x, i);
            f += self.y[i] * sigmoid(wx).ln() + (1.0 - self.y[i]) * (1.0 - sigmoid(wx)).ln();            
        }
        
        Ok(-f)
    }

    fn gradient(&self, w: &Self::Param) -> Result<Self::Param, Error> {
        let (n, p) = self.x.shape();
        let mut g = vec![0f64; w.len()];

        for i in 0..n {
            let wx = dot(w, &self.x, i);

            let dyi = sigmoid(wx) - self.y[i];
            for j in 0..p {
                g[j] += dyi * self.x[(i, j)];
            }
            g[p] += dyi;
        }
        Ok(g)
    }    
}

fn sigmoid(v: f64) -> f64 {
    if v < -40. {
        0.
    } else if v > 40. {
        1.
    } else {
        1. / (1. + f64::exp(-v))
    }
}

fn dot(w: &Vec<f64>, x: &DMatrix<f64>, m_row: usize) -> f64 {
    let mut sum = 0f64;
    let (_, p) = x.shape();
    for i in 0..p {
        sum += x[(m_row, i)] * w[i];
    }

    sum + w[p]
}

fn optimize(x: &DMatrix<f64>, y: &DVector<f64>) -> Result<(DVector<f64>, f64), Error> {      

    let (_, p) = x.shape();

    // Define cost function
    let cost = BinaryObjectiveFunction { x, y };

    // Define initial parameter vector
    let init_param: Vec<f64> = vec![0f64; p + 1];
    
    // Set condition
    let cond = ArmijoCondition::new(0.5)?;

    // set up a line search
    let linesearch = BacktrackingLineSearch::new(cond).rho(0.9)?;

    // Set up solver
    let solver = LBFGS::new(linesearch, 7);

    // Run solver
    let res = Executor::new(cost, solver, init_param)
        // .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(100)
        .run()?;

    let w = DVector::from_row_slice(&res.state().best_param);
        
    Ok((w.rows(0, 30).into_owned(), w[30]))
}

fn predict(x: &DMatrix<f64>, coeff: &DVector<f64>, intercept: f64) -> DVector<f64> {
    let mut y_hat = (x * coeff).add_scalar(intercept);
    y_hat.apply(|v| {
        if sigmoid(v) > 0.5 {
            1.0
        } else {
            0.0
        }
    });
    println!("{}", y_hat);
    y_hat.into_owned()
}

fn main() {    
    let file = File::open("creditcard.csv").unwrap();
    let credit: DMatrix<f64> = parse_csv(BufReader::new(file)).unwrap(); 
    
    let x = credit.columns(0, 30).into_owned();
    let y = credit.column(30).into_owned(); 

    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y.transpose(), 0.2, true);

    let (coeff, intercept) = optimize(&x_train, &y_train.transpose()).unwrap();

    println!("{}", coeff);
    println!("{}", intercept);

    let y_hat = predict(&x_test, &coeff, intercept);

    println!("{}", roc_auc_score(&y_test, &y_hat.transpose()));
}