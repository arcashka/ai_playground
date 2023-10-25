use csv::Reader;
use std::fs::File;

use linalg::ndarray::NDArray1;
use linalg::ndarray::NDArray2;
use linalg::LinalgError;

#[derive(Clone, Debug)]
pub struct TrainingData<T> {
    pub x: NDArray2<T>,
    pub y: NDArray1<T>,
}

#[derive(Debug)]
pub enum TrainingDataError {
    CantOpenFileError,
    InvalidFileFormatError,
    OutOfMemory,
}

impl From<LinalgError> for TrainingDataError {
    fn from(error: LinalgError) -> Self {
        match error {
            LinalgError::AllocationError => Self::OutOfMemory,
            LinalgError::DimensionMismatch => Self::InvalidFileFormatError,
        }
    }
}

pub fn read_data<T>(file: &str) -> Result<TrainingData<T>, TrainingDataError>
where
    T: num_traits::Float + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Debug,
{
    let file = File::open(file).map_err(|_| TrainingDataError::CantOpenFileError)?;

    let mut reader = Reader::from_reader(file);

    let mut y_values: Vec<T> = Vec::new();
    let mut x: Vec<T> = Vec::new();
    let mut row_len: Option<usize> = None;

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
        if row_len.is_some() {
            assert_eq!(row_len.unwrap(), row.len());
        } else {
            row_len = Some(row.len());
        }
        x.append(&mut row);
        y_values.push(y);
    }

    let y_len = y_values.len();
    let row_len = row_len.ok_or(TrainingDataError::InvalidFileFormatError)?;
    Ok(TrainingData {
        x: NDArray2::from_vec(x, [row_len, y_len])?,
        y: NDArray1::from_vec(y_values, [y_len])?,
    })
}
