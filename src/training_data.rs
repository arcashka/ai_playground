use csv::Reader;
use std::error::Error;
use std::fs::File;

use nalgebra::{DMatrix, DVector};

use crate::linear_regression::RealNumber;

pub struct TrainingData<T> {
    pub x: DMatrix<T>,
    pub y: DVector<T>,
}

pub fn read_data<T>(file: &str) -> Result<TrainingData<T>, Box<dyn Error>>
where
    T: RealNumber,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let file = File::open(file)?;

    let mut reader = Reader::from_reader(file);

    let mut number_rows: usize = 0;
    let mut number_cols: usize = 0;

    let mut y_values = Vec::<T>::new();
    let mut x_values = Vec::<T>::new();
    let one: T = num::cast(1.0).ok_or("failed cast to T")?;
    for record in reader.records() {
        let record = record?;
        let n = record.len();
        std::iter::once(Ok(one))
            .chain(record.iter().take(n - 1).map(|x_record| {
                x_record
                    .parse::<T>()
                    .map_err(|e| format!("Failed to parse x_record: {:?}", e))
            }))
            .collect::<Result<Vec<_>, String>>()?
            .into_iter()
            .for_each(|e| x_values.push(e));
        record
            .get(n - 1)
            .ok_or("Failed to get y record")?
            .parse::<T>()
            .map_err(|e| format!("Failed to parse y record: {:?}", e))
            .map(|y| y_values.push(y))?;
        number_rows += 1;
        number_cols = n;
    }
    Ok(TrainingData {
        x: DMatrix::from_row_slice(number_rows, number_cols, &x_values),
        y: DVector::from_vec(y_values),
    })
}
