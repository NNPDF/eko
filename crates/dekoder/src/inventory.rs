//! Assets manager.
use glob::glob;
use lz4_flex::frame::FrameDecoder;
use ndarray::Array4;
use ndarray_npy::NpzReader;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::{read_to_string, File};
use std::io::{BufReader, Cursor};
use std::path::PathBuf;
use yaml_rust2::{Yaml, YamlLoader};

use crate::{EKOError, Operator, Result};

/// Headers are in yaml files.
const HEADER_EXT: &str = "*.yaml";

/// Assets manager.
pub(crate) struct Inventory<K: Eq + for<'a> TryFrom<&'a Yaml, Error = EKOError>> {
    /// Working directory
    pub(crate) path: PathBuf,
    /// Available keys
    pub(crate) keys: HashMap<OsString, K>,
}

impl<K: Eq + for<'a> TryFrom<&'a Yaml, Error = EKOError>> Inventory<K> {
    /// Load all available entries.
    pub fn load_keys(&mut self) -> Result<()> {
        let path = self.path.join(HEADER_EXT);
        let path = path
            .to_str()
            .ok_or(EKOError::KeyError("due to invalid path".to_owned()))?;
        for entry in glob(path)
            .map_err(|_| EKOError::KeyError("because failed to read glob pattern".to_owned()))?
            .filter_map(core::result::Result::ok)
        {
            let cnt = YamlLoader::load_from_str(&read_to_string(&entry)?)
                .map_err(|_| EKOError::KeyError("because failed to read yaml file.".to_owned()))?;
            self.keys.insert(
                entry
                    .file_name()
                    .ok_or(EKOError::KeyError(
                        "because failed to read file name".to_owned(),
                    ))?
                    .to_os_string(),
                K::try_from(&cnt[0])?,
            );
        }
        Ok(())
    }

    /// List available keys.
    pub fn keys(&self) -> Vec<&K> {
        let mut ks = Vec::new();
        for k in self.keys.values() {
            ks.push(k);
        }
        ks
    }

    /// Check if `k` is available (with given precision).
    pub fn has(&self, k: &K) -> bool {
        self.keys.iter().any(|it| (it.1).eq(k))
    }

    /// Load `k` from disk.
    pub fn load(&self, k: &K, v: &mut Operator) -> Result<()> {
        // Find key
        let k = self.keys.iter().find(|it| (it.1).eq(k));
        let k = k.ok_or(EKOError::KeyError("because it was not found".to_owned()))?;
        let p = self.path.join(k.0).with_extension("npz.lz4");
        // Read npz.lz4
        let mut reader = BufReader::new(FrameDecoder::new(BufReader::new(File::open(&p)?)));
        let mut buffer = Vec::new();
        std::io::copy(&mut reader, &mut buffer)?;
        let mut npz = NpzReader::new(Cursor::new(buffer))
            .map_err(|_| EKOError::OperatorLoadError(p.to_owned()))?;
        let operator: Array4<f64> = npz
            .by_name("operator.npy")
            .map_err(|_| EKOError::OperatorLoadError(p.to_owned()))?;
        v.op = Some(operator);
        Ok(())
    }
}
