use ai::data_reader::cvs_reader;

fn main() {
    let training_data = cvs_reader::read_data::<f64>("resources/3.csv").unwrap();
    println!("X: {:?}", training_data.x);
    println!("Y: {:?}", training_data.y);
}
