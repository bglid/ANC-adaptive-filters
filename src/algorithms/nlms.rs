use crate::types::{FilterWeights, OutputSample, SampleBuffer};
use crate::{Error, Result};

use crate::algorithms::Algorithm;

#[derive(Debug, Clone)]
#[allow(clippy::exhaustive_structs, reason = "No more fields have to be added")]
pub struct NormalizedLeastMeanSquares {
    mu: f64,
    eps: f64,
}
impl NormalizedLeastMeanSquares {
    /// # Errors
    ///
    /// Returns an error if mu <= 0.0.
    pub fn new(mu: f64, eps: f64) -> Result<Self> {
        if mu > 0.0 {
            Ok(NormalizedLeastMeanSquares { mu, eps })
        } else {
            Err(Error::NonPositiveStepSize)
        }
    }
}
impl Algorithm for NormalizedLeastMeanSquares {
    fn update_step(
        &self,
        weights: &mut FilterWeights,
        error: OutputSample,
        noise_ref: &SampleBuffer,
    ) {
        let norm_squared: f64 = noise_ref.iter().map(|x| x * x).sum();

        for (w, x) in weights.iter_mut().zip(noise_ref.iter()) {
            *w += (self.mu / (self.eps + norm_squared)) * (*error) * x;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::indexing_slicing, reason = "Tests")]
mod tests {
    use super::*;
    use crate::{
        test_utils::{approx_equal, sample_buffer_from},
        types::FilterWeights,
    };
    use std::num::NonZero;

    #[test]
    fn update_nlms_1() {
        let nlms = NormalizedLeastMeanSquares::new(0.5, 1e-8).unwrap();
        let e_n = OutputSample(2.0);
        let x_n = sample_buffer_from(&[1.0, -1.0]);
        let expected = [1.0 / (2.0 + nlms.eps), -1.0 / (2.0 + nlms.eps)];
        let mut weights = FilterWeights::zeros(NonZero::new(2).unwrap());

        nlms.update_step(&mut weights, e_n, &x_n);

        let output_correct = weights
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| approx_equal(*a, *b, 1e-6));
        assert!(output_correct);
    }

    #[test]
    fn update_nlms_2() {
        let nlms = NormalizedLeastMeanSquares::new(1.0, 1e-8).unwrap();
        let e_n = OutputSample(1.0);
        let x_n = sample_buffer_from(&[5.0, 2.0]);
        let expected = [(5.0 / (29.0 + nlms.eps)), (2.0 / (29.0 + nlms.eps))];
        let mut weights = FilterWeights::zeros(NonZero::new(2).unwrap());

        nlms.update_step(&mut weights, e_n, &x_n);

        let output_correct = weights
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| approx_equal(*a, *b, 1e-6));
        assert!(output_correct);
    }

    #[test]
    fn mu_range() {
        NormalizedLeastMeanSquares::new(1.0, 1e-8).unwrap();
        NormalizedLeastMeanSquares::new(f64::MAX, 1e-8).unwrap();

        assert!(matches!(
            NormalizedLeastMeanSquares::new(0.0, 1e-8),
            Err(Error::NonPositiveStepSize)
        ));
        assert!(matches!(
            NormalizedLeastMeanSquares::new(-1.0, 1e-8),
            Err(Error::NonPositiveStepSize)
        ));
    }
}
