//! eko output interface.
use float_cmp::approx_eq;
use hashbrown::HashMap;
use lz4_flex::frame::FrameDecoder;
use ndarray::Array4;
use ndarray_npy::NpzReader;
use std::fs::remove_dir_all;
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor};
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
    #[error("No target path given")]
    NoTargetPath,
    #[error("Target path `{0}` already exists")]
    TargetAlreadyExists(PathBuf),
    #[error("Loading operator from `{0}` failed")]
    OperatorLoadError(PathBuf),
    #[error("Failed to read key(s) `{0}`")]
    KeyError(String),
}

/// My result type has always my errros.
type Result<T> = std::result::Result<T, EKOError>;

/// A reference point in the evolution atlas.
pub struct EvolutionPoint {
    /// Evolution scale.
    pub scale: f64,
    /// Number of flavors
    pub nf: i64,
}

impl inventory::HeaderT for EvolutionPoint {
    /// Load from yaml.
    fn load_from_yaml(yml: &Yaml) -> Result<Box<Self>> {
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
        Ok(Box::new(Self { scale, nf }))
    }

    /// Comparator.
    fn eq(&self, other: &Self, ulps: i64) -> bool {
        self.nf == other.nf && approx_eq!(f64, self.scale, other.scale, ulps = ulps)
    }
}

/// 4D evolution operator.
pub struct Operator {
    pub op: Array4<f64>,
}

impl inventory::ValueT for Operator {
    const FILE_SUFFIX: &'static str = "npz.lz4";
    fn load_from_path(&mut self, p: PathBuf) -> Result<()> {
        let mut reader =
            BufReader::new(FrameDecoder::new(BufReader::new(File::open(p.to_owned())?)));
        let mut buffer = Vec::new();
        std::io::copy(&mut reader, &mut buffer)?;
        let mut npz = NpzReader::new(Cursor::new(buffer))
            .map_err(|_| EKOError::OperatorLoadError(p.to_owned()))?;
        let operator: Array4<f64> = npz
            .by_name("operator.npy")
            .map_err(|_| EKOError::OperatorLoadError(p.to_owned()))?;
        self.op = operator;
        Ok(())
    }
}

impl Operator {
    /// Empty initializer.
    pub fn zeros() -> Self {
        Self {
            op: Array4::zeros((0, 0, 0, 0)),
        }
    }
}

/// EKO output
pub struct EKO {
    /// Working directory
    path: PathBuf,
    /// Associated archive path
    tar_path: Option<PathBuf>,
    /// allow content modifications?
    read_only: bool,
    /// final operators
    operators: inventory::Inventory<EvolutionPoint>,
}

/// Operators directory.
const DIR_OPERATORS: &'static str = "operators/";

impl EKO {
    /// Check our working directory is safe.
    fn check(&self) -> Result<()> {
        let path_exists = self.path.try_exists().is_ok_and(|x| x);
        if !path_exists {
            return Err(EKOError::NoWorkingDir);
        }
        Ok(())
    }

    /// Remove the working directory.
    fn destroy(&self) -> Result<()> {
        self.check()?;
        Ok(remove_dir_all(self.path.to_owned())?)
    }

    /// Write content back to an archive and destroy working directory.
    pub fn close(&self, allow_overwrite: bool) -> Result<()> {
        self.write(allow_overwrite, true)
    }

    /// Write content back to an archive.
    pub fn write(&self, allow_overwrite: bool, destroy: bool) -> Result<()> {
        self.check()?;
        // in read-only there is nothing to do then to destroy, since we couldn't
        if self.read_only && destroy {
            return self.destroy();
        }
        // check we can write
        if self.tar_path.is_none() {
            return Err(EKOError::NoTargetPath);
        }
        let dst = self.tar_path.to_owned().ok_or(EKOError::NoTargetPath)?;
        let dst_exists = dst.try_exists().is_ok_and(|x| x);
        if !allow_overwrite && dst_exists {
            return Err(EKOError::TargetAlreadyExists(dst));
        }
        // create writer
        let dst_file = File::create(dst.to_owned())?;
        let dst_file = BufWriter::with_capacity(128 * 1024, dst_file);
        let mut ar = tar::Builder::new(dst_file);
        // do it!
        ar.append_dir_all(".", self.path.to_owned())?;
        // cleanup
        if destroy {
            self.destroy()?;
        }
        Ok(())
    }

    /// Set the archive path.
    pub fn set_tar_path(&mut self, tar_path: PathBuf) {
        self.tar_path = Some(tar_path.to_owned());
    }

    /// Open tar from `src` to `dst` for reading.
    pub fn read(src: PathBuf, dst: PathBuf) -> Result<Self> {
        Self::extract(src, dst, true)
    }

    /// Open tar from `src` to `dst` for editing.
    pub fn edit(src: PathBuf, dst: PathBuf) -> Result<Self> {
        Self::extract(src, dst, false)
    }

    /// Extract tar file from `src` to `dst`.
    pub fn extract(src: PathBuf, dst: PathBuf, read_only: bool) -> Result<Self> {
        let mut ar = tar::Archive::new(File::open(src.to_owned())?);
        ar.unpack(dst.to_owned())?;
        let mut obj = Self::load_opened(dst, read_only)?;
        obj.set_tar_path(src);
        Ok(obj)
    }

    /// Load an EKO from a directory `path` (instead of tar).
    pub fn load_opened(path: PathBuf, read_only: bool) -> Result<Self> {
        let mut operators = inventory::Inventory {
            path: path.join(DIR_OPERATORS),
            keys: HashMap::new(),
        };
        operators.load_keys()?;
        Ok(Self {
            path,
            tar_path: None,
            read_only,
            operators,
        })
    }

    /// List available evolution points.
    pub fn available_operators(&self) -> Vec<&Box<EvolutionPoint>> {
        self.operators.keys()
    }

    /// Check if the operator at the evolution point `ep` is available.
    pub fn has_operator(&self, ep: &EvolutionPoint, ulps: i64) -> bool {
        self.operators.has(ep, ulps)
    }

    /// Load the operator at the evolution point `ep` from disk.
    pub fn load_operator(
        &mut self,
        ep: &EvolutionPoint,
        ulps: i64,
        op: &mut Operator,
    ) -> Result<()> {
        self.operators.load(ep, ulps, op)?;
        Ok(())
    }
}