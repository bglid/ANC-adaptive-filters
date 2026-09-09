#![allow(
    clippy::multiple_inherent_impl,
    reason = "Defining methods that are only needed for this module and shouldn't be compiled otherwise."
)]

use pyo3::prelude::*;

use pyo3::exceptions::PyValueError;

use numpy::{PyArray1, PyReadonlyArray1};

use crate::Error;
use crate::algorithms::{Algorithm, LeastMeanSquares, NormalizedLeastMeanSquares};
use crate::filters::FilterBase;
use crate::py_utils::generate_filter_bindings;
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

#[allow(
    clippy::needless_pass_by_value,
    reason = "PyArrays must be passed by value"
)]
// Because the wrappers for adapt() and filter() would only differ in one line,
// we use this underlying implementation.
fn adapt_filter_impl<'py, A>(
    filter: &mut FilterBase<A>,
    py: Python<'py>,
    input_signal: PyReadonlyArray1<f64>,
    noise_ref: PyReadonlyArray1<f64>,
    op: FilterOperation,
) -> PyResult<Bound<'py, PyArray1<f64>>>
where
    A: Algorithm,
{
    let input_signal = InputSignal::from_pyarray(&input_signal)?;
    let noise_ref = NoiseReference::from_pyarray(&noise_ref)?;

    let output_signal = match op {
        FilterOperation::Adapt => filter.adapt(&input_signal, &noise_ref),
        FilterOperation::Filter => filter.filter(&input_signal, &noise_ref),
    }
    .map_err(|e| e.to_pyerr())?;

    Ok(PyArray1::from_vec(py, output_signal))
}

#[pymodule]
mod adaptif {
    #[pymodule_export]
    use super::{LMSFilter, NLMSFilter};
}

// ------- LMS
#[pyclass]
pub struct LMSFilter(FilterBase<LeastMeanSquares>);
#[pymethods]
impl LMSFilter {
    #[new]
    fn new(mu: f64, window_size: usize) -> PyResult<Self> {
        let lms = LeastMeanSquares::new(mu).map_err(|e| e.to_pyerr())?;
        match FilterBase::<LeastMeanSquares>::new(lms, window_size) {
            Some(filter) => Ok(Self(filter)),
            None => Err(PyValueError::new_err("window_size cannot be zero")),
        }
    }
}

generate_filter_bindings!(LMSFilter);

// ------- NLMS
#[pyclass]
pub struct NLMSFilter(FilterBase<NormalizedLeastMeanSquares>);
#[pymethods]
impl NLMSFilter {
    #[new]
    fn new(mu: f64, window_size: usize) -> PyResult<Self> {
        let nlms = NormalizedLeastMeanSquares::new(mu, 1e-8).map_err(|e| e.to_pyerr())?;
        match FilterBase::<NormalizedLeastMeanSquares>::new(nlms, window_size) {
            Some(filter) => Ok(Self(filter)),
            None => Err(PyValueError::new_err("window_size cannot be zero")),
        }
    }
}

generate_filter_bindings!(NLMSFilter);
// -------

#[derive(Debug, Clone, Copy)]
enum FilterOperation {
    Adapt,
    Filter,
}
