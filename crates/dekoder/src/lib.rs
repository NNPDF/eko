//! eko output interface.
use float_cmp::approx_eq;
use ndarray::Array4;
use std::collections::HashMap;
use std::fs::remove_dir_all;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use thiserror::Error;
use yaml_rust2::Yaml;

mod inventory;

/// The EKO errors.
#[derive(Error, Debug)]
pub enum EKOError {
    #[error("No working directory")]
    NoWorkingDir,
    #[error("I/O error")]
    IOError(#[from] std::io::Error),
    #[error("Loading operator from `{0}` failed")]
    OperatorLoadError(PathBuf),
    #[error("Failed to read key(s) `{0}`")]
    KeyError(String),
}

/// My result type has always my errros.
type Result<T> = std::result::Result<T, EKOError>;

/// Default value for the float comparison inside `EvolutionPoint`.
const EP_CMP_ULPS: i64 = 64;

/// A reference point in the evolution atlas.
pub struct EvolutionPoint {
    /// Evolution scale.
    pub scale: f64,
    /// Number of flavors
    pub nf: i64,
}

/// 4D evolution operator.
pub struct Operator {
    pub op: Option<Array4<f64>>,
}

impl Default for Operator {
    /// Empty initializer.
    fn default() -> Self {
        Self { op: None }
    }
}

impl TryFrom<&Yaml> for EvolutionPoint {
    type Error = EKOError;

    /// Load from yaml.
    fn try_from(yml: &Yaml) -> Result<Self> {
        // work around float representation
        let scale = yml["scale"].as_f64();
        let scale = if scale.is_some() {
            scale.ok_or(EKOError::KeyError(
                "because failed to read scale as float".to_owned(),
            ))?
        } else {
            yml["scale"].as_i64().ok_or(EKOError::KeyError(
                "because failed to read scale as float from int".to_owned(),
            ))? as f64
        };
        let nf = yml["nf"]
            .as_i64()
            .ok_or(EKOError::KeyError("because failed to read nf".to_owned()))?;
        Ok(Self { scale, nf })
    }
}

impl PartialEq for EvolutionPoint {
    /// (Protected) comparator.
    fn eq(&self, other: &Self) -> bool {
        self.nf == other.nf && approx_eq!(f64, self.scale, other.scale, ulps = EP_CMP_ULPS)
    }
}

impl Eq for EvolutionPoint {}

/// EKO output
pub struct EKO {
    /// Working directory
    path: PathBuf,
    /// final operators
    operators: inventory::Inventory<EvolutionPoint>,
}

/// Operators directory.
const DIR_OPERATORS: &str = "operators/";
/// Buffer capacity for tar writer
const TAR_WRITER_CAPACITY: usize = 128 * 1024;

impl EKO {
    /// Check our working directory is safe.
    fn assert_working_dir(&self) -> Result<()> {
        if !self.path.try_exists().is_ok_and(|x| x) {
            return Err(EKOError::NoWorkingDir);
        }
        Ok(())
    }

    /// Remove the working directory.
    pub fn destroy(&self) -> Result<()> {
        self.assert_working_dir()?;
        Ok(remove_dir_all(&self.path)?)
    }

    /// Write content back to an archive and destroy working directory.
    pub fn write_and_destroy(&self, dst: PathBuf) -> Result<()> {
        self.write(dst)?;
        self.destroy()
    }

    /// Write content back to an archive.
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
        let mut operators = inventory::Inventory {
            path: path.join(DIR_OPERATORS),
            keys: HashMap::new(),
        };
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
    pub fn load_operator(&self, ep: &EvolutionPoint, op: &mut Operator) -> Result<()> {
        self.assert_working_dir()?;
        self.operators.load(ep, op)
    }
}
