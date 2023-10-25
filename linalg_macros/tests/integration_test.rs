#[cfg(test)]
mod tests {
    use linalg_macros::ndarray;

    #[test]
    fn macro_test() {
        let a = ndarray!([
            [[5.2, 1.2], [2.2, 3.3], [4.0, 52.0]],
            [[5.2, 1.2], [2.2, 3.3], [4.0, 56.0]]
        ]);
        assert_eq!(a.get_element([0, 0, 1]), 1.2);
        assert_eq!(a.get_element([0, 1, 0]), 2.2);
        assert_eq!(a.get_element([1, 2, 1]), 56.0);
    }
}
