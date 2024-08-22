//! Utilities for reading and writing an eko output.
use std::fs::remove_dir_all;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use yaml_rust2::Yaml;

use crate::{EKOError, Operator, Result};

/// Default rel. error for the float comparison inside `EvolutionPoint`.
const EP_CMP_RTOL: f64 = 1e-5;
/// Default abs. error for the float comparison inside `EvolutionPoint`.
const EP_CMP_ATOL: f64 = 1e-3;

/// A reference point in the evolution atlas.
pub struct EvolutionPoint {
    /// Evolution scale.
    pub scale: f64,
    /// Number of flavors
    pub nf: i64,
}

impl TryFrom<&Yaml> for EvolutionPoint {
    type Error = EKOError;

    /// Load from yaml.
    fn try_from(yml: &Yaml) -> Result<Self> {
        // work around float representation
        let scale = yml["scale"].as_f64();
        let scale = match scale {
            Some(scale) => scale,
            None => yml["scale"].as_i64().ok_or(EKOError::KeyError(
                "because failed to read scale as float from int".to_owned(),
            ))? as f64,
        };
        let nf = yml["nf"]
            .as_i64()
            .ok_or(EKOError::KeyError("because failed to read nf".to_owned()))?;
        Ok(Self { scale, nf })
    }
}

/// Reimplementation of [`np.isclose`](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html#numpy-isclose).
fn is_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

impl PartialEq for EvolutionPoint {
    /// Comparator using default tolerance for float comparisons.
    fn eq(&self, other: &Self) -> bool {
        self.nf == other.nf && is_close(self.scale, other.scale, EP_CMP_RTOL, EP_CMP_ATOL)
    }
}

impl Eq for EvolutionPoint {}

/// EKO output
pub struct EKO {
    /// Working directory
    path: PathBuf,
    /// final operators
    operators: crate::inventory::Inventory<EvolutionPoint>,
}

/// Operators directory.
const DIR_OPERATORS: &str = "operators/";
/// Buffer capacity for tar writer
const TAR_WRITER_CAPACITY: usize = 128 * 1024;

impl EKO {
    /// Check our working directory is safe.
    fn assert_working_dir(&self) -> Result<()> {
        self.path
            .exists()
            .then_some(())
            .ok_or(EKOError::NoWorkingDir)
    }

    /// Remove the working directory.
    pub fn destroy(&self) -> Result<()> {
        self.assert_working_dir()?;
        Ok(remove_dir_all(&self.path)?)
    }

    /// Write the content to an archive `dst` and remove the working directory.
    pub fn write_and_destroy(&self, dst: PathBuf) -> Result<()> {
        self.write(dst)?;
        self.destroy()
    }

    /// Write the content to an archive `dst`.
    pub fn write(&self, dst: PathBuf) -> Result<()> {
        self.assert_working_dir()?;
        // create writer
        let dst_file = File::create(&dst)?;
        let dst_file = BufWriter::with_capacity(TAR_WRITER_CAPACITY, dst_file);
        let mut ar = tar::Builder::new(dst_file);
        // do it!
        Ok(ar.append_dir_all(".", &self.path)?)
    }

    /// Extract tar file from `src` to `dst`.
    pub fn extract(src: PathBuf, dst: PathBuf) -> Result<Self> {
        let mut ar = tar::Archive::new(File::open(&src)?);
        ar.unpack(&dst)?;
        Self::load_opened(dst)
    }

    /// Load an EKO from a directory `path` (instead of tar).
    pub fn load_opened(path: PathBuf) -> Result<Self> {
        let mut operators = crate::inventory::Inventory::new(path.join(DIR_OPERATORS));
        operators.load_keys()?;
        let obj = Self { path, operators };
        obj.assert_working_dir()?;
        Ok(obj)
    }

    /// List available evolution points.
    pub fn available_operators(&self) -> Vec<&EvolutionPoint> {
        self.operators.keys()
    }

    /// Check if the operator at the evolution point `ep` is available.
    pub fn has_operator(&self, ep: &EvolutionPoint) -> bool {
        self.operators.has(ep)
    }

    /// Load the operator at the evolution point `ep` from disk.
    pub fn load_operator(&self, ep: &EvolutionPoint) -> Result<Operator> {
        self.assert_working_dir()?;
        self.operators.load(ep)
    }
}
