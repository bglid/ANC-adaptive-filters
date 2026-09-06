use crate::types::{FilterWeights, OutputSample, SampleBuffer};

mod lms;
mod nlms;

pub use lms::LeastMeanSquares;
pub use nlms::NormalizedLeastMeanSquares;

/// Trait used for implementing algorithms used in conjuction with `FilterBase`.
pub trait Algorithm {
    /// Updates the weights for the next time step based on the algorithm's update rules.
    /// This function is called every processing iteration by the filter during adapation.
    /// `error` is the cleaned sample from the current time step.
    /// `noise_ref` is the noise reference signal within the current processing window (the $k$ most recent samples).
    ///
    fn update_step(
        &self,
        weights: &mut FilterWeights,
        error: OutputSample,
        noise_ref: &SampleBuffer,
    );
}
