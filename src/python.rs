#![allow(
    clippy::multiple_inherent_impl,
    reason = "Defining methods that are only needed for this module and shouldn't be compiled otherwise."
)]

use pyo3::prelude::*;

use pyo3::exceptions::PyValueError;

use numpy::{PyArray1, PyReadonlyArray1};

use crate::Error;
use crate::algorithms::LeastMeanSquares;
use crate::filters::LMSFilter as RustLMSFilter;
use crate::types::{InputSignal, NoiseReference};

impl Error {
    fn to_pyerr(&self) -> PyErr {
        match *self {
            Self::EmptyInputArr | Self::NoiseRefTooShort { .. } | Self::NonPositiveStepSize => {
                PyValueError::new_err(self.to_string())
            }
        }
    }
}

// In Python, we use NumPy arrays as inputs, so we have to convert them to the Rust input types.
// Because we're using slices in Rust, the input NumPy arrays need to be contiguous.
// Strided slices like x[::2] or x[:, 0] are not allowed, and need to be made contiguous first.
impl<'a> InputSignal<'a> {
    fn from_pyarray(input_signal: &'a PyReadonlyArray1<f64>) -> PyResult<InputSignal<'a>> {
        let input_signal = input_signal.as_slice().map_err(|_e| {
            PyValueError::new_err(
                "input_signal must be a contiguous NumPy array; use numpy.ascontiguousarray().",
            )
        })?;

        let input_signal = InputSignal::new(input_signal).map_err(|e| e.to_pyerr())?;
        Ok(input_signal)
    }
}
impl<'a> NoiseReference<'a> {
    fn from_pyarray(noise_ref: &'a PyReadonlyArray1<f64>) -> PyResult<NoiseReference<'a>> {
        let noise_ref = noise_ref.as_slice().map_err(|_e| {
            PyValueError::new_err(
                "noise_ref must be a contiguous NumPy array; use numpy.ascontiguousarray().",
            )
        })?;
        let noise_ref = NoiseReference::new(noise_ref).map_err(|e| e.to_pyerr())?;
        Ok(noise_ref)
    }
}

#[pymodule]
mod adaptif {
    #[pymodule_export]
    use super::LMSFilter;
}

/// Declarative macro for generating filter binding setup.
macro_rules! pyo3_filter {
    ($name: ident, $filter: ident, $algo: ident) => {
        #[pyclass]
        pub struct $name($filter);
        #[pymethods]
        impl $name {
            #[new]
            fn new(mu: f64, window_size: usize) -> PyResult<Self> {
                let filter = $algo::new(mu).map_err(|e| e.to_pyerr())?;
                match $filter::new(filter, window_size) {
                    Some(filter) => Ok(Self(filter)),
                    None => Err(PyValueError::new_err("window_size cannot be zero")),
                }
            }

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
                self.adapt_filter_impl(py, input_signal, noise_ref, FilterOperation::Adapt)
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
                self.adapt_filter_impl(py, input_signal, noise_ref, FilterOperation::Filter)
            }
        }

        // The methods below won't be exported
        impl $name {
            #[allow(
                clippy::needless_pass_by_value,
                reason = "PyArrays must be passed by value"
            )]
            // Because the wrappers for adapt() and filter() would only differ in one line,
            // we use this underlying implementation.
            fn adapt_filter_impl<'py>(
                &mut self,
                py: Python<'py>,
                input_signal: PyReadonlyArray1<f64>,
                noise_ref: PyReadonlyArray1<f64>,
                op: FilterOperation,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                let input_signal = InputSignal::from_pyarray(&input_signal)?;
                let noise_ref = NoiseReference::from_pyarray(&noise_ref)?;

                let output_signal = match op {
                    FilterOperation::Adapt => self.0.adapt(&input_signal, &noise_ref),
                    FilterOperation::Filter => self.0.filter(&input_signal, &noise_ref),
                }
                .map_err(|e| e.to_pyerr())?;

                Ok(PyArray1::from_vec(py, output_signal))
            }
        }
    };
}
pyo3_filter!(LMSFilter, RustLMSFilter, LeastMeanSquares);

// #[pyclass]
// pub struct LMSFilter(RustLMSFilter);
// #[pymethods]
// impl LMSFilter {
//     #[new]
//     fn new(mu: f64, window_size: usize) -> PyResult<Self> {
//         let lms = LeastMeanSquares::new(mu).map_err(|e| e.to_pyerr())?;
//         match RustLMSFilter::new(lms, window_size) {
//             Some(filter) => Ok(Self(filter)),
//             None => Err(PyValueError::new_err("window_size cannot be zero")),
//         }
//     }
//
//     #[getter]
//     fn window_size(&self) -> usize {
//         self.0.window_size()
//     }
//
//     // TODO: weights() (+ check before/after in tests)
//
//     fn adapt<'py>(
//         &mut self,
//         py: Python<'py>,
//         input_signal: PyReadonlyArray1<f64>,
//         noise_ref: PyReadonlyArray1<f64>,
//     ) -> PyResult<Bound<'py, PyArray1<f64>>> {
//         self.adapt_filter_impl(py, input_signal, noise_ref, FilterOperation::Adapt)
//     }
//
//     fn filter<'py>(
//         // self has to be mutable so that we can call adapt_filter_impl().
//         // In Python there is no immutability, and we later pass an immutable reference
//         // to the Rust filter() fn with the actual implementation, so this is fine.
//         &mut self,
//         py: Python<'py>,
//         input_signal: PyReadonlyArray1<f64>,
//         noise_ref: PyReadonlyArray1<f64>,
//     ) -> PyResult<Bound<'py, PyArray1<f64>>> {
//         self.adapt_filter_impl(py, input_signal, noise_ref, FilterOperation::Filter)
//     }
// }

// The methods below won't be exported
// impl LMSFilter {
//     #[allow(
//         clippy::needless_pass_by_value,
//         reason = "PyArrays must be passed by value"
//     )]
//     // Because the wrappers for adapt() and filter() would only differ in one line,
//     // we use this underlying implementation.
//     fn adapt_filter_impl<'py>(
//         &mut self,
//         py: Python<'py>,
//         input_signal: PyReadonlyArray1<f64>,
//         noise_ref: PyReadonlyArray1<f64>,
//         op: FilterOperation,
//     ) -> PyResult<Bound<'py, PyArray1<f64>>> {
//         let input_signal = InputSignal::from_pyarray(&input_signal)?;
//         let noise_ref = NoiseReference::from_pyarray(&noise_ref)?;
//
//         let output_signal = match op {
//             FilterOperation::Adapt => self.0.adapt(&input_signal, &noise_ref),
//             FilterOperation::Filter => self.0.filter(&input_signal, &noise_ref),
//         }
//         .map_err(|e| e.to_pyerr())?;
//
//         Ok(PyArray1::from_vec(py, output_signal))
//     }
// }

#[derive(Debug, Clone, Copy)]
enum FilterOperation {
    Adapt,
    Filter,
}
