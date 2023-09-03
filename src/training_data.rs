use csv::Reader;
use std::error::Error;
use std::fs::File;

use nalgebra::{DMatrix, DVector};

pub struct TrainingData<T> {
    pub x: DMatrix<T>,
    pub y: DVector<T>,
}

pub fn read_data<T>(file: &str) -> Result<TrainingData<T>, Box<dyn Error>>
where
    T: nalgebra::RealField + num::NumCast + std::str::FromStr + std::default::Default + Copy,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let file = File::open(file)?;

    let mut reader = Reader::from_reader(file);

    let mut number_rows: usize = 0;
    let mut number_cols: usize = 0;

    let mut y_values = Vec::<T>::new();
    let mut x_values = Vec::<T>::new();
    let one: T = num::cast(1.0).unwrap();
    for record in reader.records() {
        let record = record?;
        let n = record.len();
        std::iter::once(one)
            .chain(
                record
                    .iter()
                    .take(n - 1)
                    .map(|x_record| x_record.parse::<T>().unwrap_or_default()),
            )
            .for_each(|e| x_values.push(e));
        y_values.push(record.get(n - 1).unwrap().parse::<T>().unwrap());
        number_rows += 1;
        number_cols = n;
    }
    Ok(TrainingData {
        x: DMatrix::from_row_slice(number_rows, number_cols, &x_values),
        y: DVector::from_vec(y_values),
    })
}
