use csv::Reader;
use std::error::Error;
use std::fs::File;

pub struct TrainingExample {
    pub x: Vec<f64>,
    pub y: f64,
}

pub struct TrainingData {
    pub x_count: usize,
    pub examples: Vec<TrainingExample>,
}

impl TrainingData {
    fn new() -> Self {
        TrainingData {
            x_count: 0,
            examples: Vec::new(), // Here we initialize the Vec
        }
    }
}

pub fn read_data(file: &str) -> Result<TrainingData, Box<dyn Error>> {
    let file = File::open(file)?;

    let mut reader = Reader::from_reader(file);
    let mut data = TrainingData::new();
    for record in reader.records() {
        let record = record?;
        let n = record.len();
        let x_values: Vec<f64> = std::iter::once(1.0)
            .chain(
                record
                    .iter()
                    .take(n - 1)
                    .map(|x_record| x_record.parse::<f64>().unwrap_or_default()),
            )
            .collect();
        let y_value = record.get(n - 1).unwrap().parse::<f64>().unwrap();
        data.x_count = x_values.len();
        data.examples.push(TrainingExample {
            x: x_values,
            y: y_value,
        });
    }
    Ok(data)
}
