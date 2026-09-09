#![allow(
    clippy::pub_with_shorthand,
    reason = "Project prefers shorthand visibility syntax"
)]
/// Declarative macro for generating filter binding setup.
macro_rules! generate_filter_bindings {
    ($name: ident) => {
        #[pymethods]
        impl $name {
            #[getter]
            fn window_size(&self) -> usize {
                self.0.window_size()
            }

            // TODO: weights() (+ check before/after in tests)

            fn adapt<'py>(
                &mut self,
                py: Python<'py>,
                input_signal: PyReadonlyArray1<f64>,
                noise_ref: PyReadonlyArray1<f64>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                adapt_filter_impl(
                    &mut self.0,
                    py,
                    input_signal,
                    noise_ref,
                    FilterOperation::Adapt,
                )
            }

            fn filter<'py>(
                // self has to be mutable so that we can call adapt_filter_impl().
                // In Python there is no immutability, and we later pass an immutable reference
                // to the Rust filter() fn with the actual implementation, so this is fine.
                &mut self,
                py: Python<'py>,
                input_signal: PyReadonlyArray1<f64>,
                noise_ref: PyReadonlyArray1<f64>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                adapt_filter_impl(
                    &mut self.0,
                    py,
                    input_signal,
                    noise_ref,
                    FilterOperation::Filter,
                )
            }
        }
    };
}

pub(crate) use generate_filter_bindings;
