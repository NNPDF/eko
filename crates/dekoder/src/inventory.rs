//! Assets manager.
use lz4_flex::frame::FrameDecoder;
use ndarray_npy::NpzReader;
use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::fs::{read_dir, read_to_string, File};
use std::io::Cursor;
use std::path::PathBuf;
use yaml_rust2::{Yaml, YamlLoader};

use crate::{EKOError, Operator, Result};

/// Headers are in yaml files.
const HEADER_EXT: &str = "yaml";

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
        for entry in read_dir(&self.path)? {
            // is header file?
            let entry = entry?.path();
            if !entry
                .extension()
                .is_some_and(|ext| ext.eq(OsStr::new(HEADER_EXT)))
            {
                continue;
            }
            // read
            let cnt = YamlLoader::load_from_str(&read_to_string(&entry)?)
                .map_err(|_| EKOError::KeyError("because failed to read yaml file.".to_owned()))?;
            // add to register
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
        self.keys.values().collect()
    }

    /// Check if `k` is available (with given precision).
    pub fn has(&self, k: &K) -> bool {
        self.keys.iter().any(|it| (it.1).eq(k))
    }

    /// Load `k` from disk.
    pub fn load(&self, k: &K, v: &mut Operator) -> Result<()> {
        // Find key
        let k = self
            .keys
            .iter()
            .find(|it| (it.1).eq(k))
            .ok_or(EKOError::KeyError("because it was not found".to_owned()))?;
        let p = self.path.join(k.0).with_extension("npz.lz4");
        // Read npz.lz4
        let mut reader = FrameDecoder::new(File::open(&p)?);
        let mut buffer = Vec::new();
        std::io::copy(&mut reader, &mut buffer)?;
        v.op = Some(
            NpzReader::new(Cursor::new(buffer))
                .map_err(|_| EKOError::OperatorLoadError(p.to_owned()))?
                .by_name("operator.npy")
                .map_err(|_| EKOError::OperatorLoadError(p.to_owned()))?,
        );
        Ok(())
    }
}
