//! Reading and writing eko outputs.
use ndarray::Array4;
use std::path::PathBuf;
use thiserror::Error;

pub mod eko;
mod inventory;

/// The EKO errors.
#[derive(Error, Debug)]
pub enum EKOError {
    /// Working directory is not usable.
    #[error("No working directory")]
    NoWorkingDir,
    /// Underlying I/O error.
    #[error("I/O error")]
    IOError(#[from] std::io::Error),
    /// 4D operator is not readable.
    #[error("Loading operator from `{0}` failed")]
    OperatorLoadError(PathBuf),
    /// Lookup error in an inventory.
    #[error("Failed to read key(s) `{0}`")]
    KeyError(String),
}

/// A specialized [`Result`] type for EKO manipulation.
///
/// [`Result`]: std::result::Result
pub type Result<T> = std::result::Result<T, EKOError>;

/// 4D evolution operator.
pub struct Operator {
    /// The actual rank 4 tensor.
    pub op: Option<Array4<f64>>,
}

impl Default for Operator {
    /// Empty initializer.
    fn default() -> Self {
        Self { op: None }
    }
}
