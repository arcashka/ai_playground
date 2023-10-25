extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Expr, ExprArray};

#[proc_macro]
pub fn ndarray(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as Expr);
    let mut flattened_elements = proc_macro2::TokenStream::new();
    let mut dimensions = Vec::new();

    flatten_array(input, &mut flattened_elements, &mut dimensions, 0);

    let dimensions_len = dimensions.len();

    let output = quote! {
        {
            use linalg::ndarray::NDArray;

            let flattened_vec = vec![#flattened_elements];
            const DIM: usize = #dimensions_len;
            const SHAPE: [usize; DIM] = [ #(#dimensions),* ];

            NDArray::<_, DIM>::from_vec(flattened_vec, SHAPE).unwrap()
        }
    };
    TokenStream::from(output)
}

fn flatten_array(
    expr: Expr,
    flattened_elements: &mut proc_macro2::TokenStream,
    dimensions: &mut Vec<usize>,
    level: usize,
) {
    if let Expr::Array(ExprArray { elems, .. }) = expr {
        let len = elems.len();
        if dimensions.len() < level + 1 {
            dimensions.push(len);
        } else {
            assert_eq!(
                dimensions[level], len,
                "All subarrays of one dimension must be same length"
            );
        }

        for inner_array in elems {
            flatten_array(inner_array, flattened_elements, dimensions, level + 1);
        }
    } else {
        flattened_elements.extend(quote! { #expr, });
    }
}
