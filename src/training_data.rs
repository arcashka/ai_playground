use csv::Reader;
use std::fs::File;

use crate::array;

#[derive(Clone)]
pub struct TrainingData<T> {
    pub x: array::Array2<T>,
    pub y: array::Array1<T>,
}

#[derive(Debug)]
pub enum TrainingDataError {
    CantOpenFileError,
    InvalidFileFormatError,
}

pub fn read_data<T>(file: &str) -> Result<TrainingData<T>, TrainingDataError>
where
    T: num_traits::Float + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let file = File::open(file).map_err(|_| TrainingDataError::CantOpenFileError)?;

    let mut reader = Reader::from_reader(file);

    let mut y_values: Vec<T> = Vec::new();
    let mut x: Option<array::Array2<T>> = None;

    for record in reader.records() {
        let record = record.map_err(|_| TrainingDataError::InvalidFileFormatError)?;
        let n = record.len();
        let row: Result<Vec<_>, _> = std::iter::once(Ok(T::one()))
            .chain(record.iter().take(n).map(|x_record| {
                x_record
                    .parse::<T>()
                    .map_err(|_| TrainingDataError::InvalidFileFormatError)
            }))
            .collect();
        let mut row = row?;
        let y = row.pop().ok_or(TrainingDataError::InvalidFileFormatError)?;
        match x.as_mut() {
            Some(x) => x
                .push_row(array::Array1::from_vec(row).view())
                .map_err(|_| TrainingDataError::InvalidFileFormatError)?,
            None => {
                x = Some(
                    array::Array2::from_shape_vec((1, row.len()), row)
                        .map_err(|_| TrainingDataError::InvalidFileFormatError)?,
                )
            }
        };
        y_values.push(y);
    }
    Ok(TrainingData {
        x: x.unwrap(),
        y: array::Array1::from_vec(y_values),
    })
}
