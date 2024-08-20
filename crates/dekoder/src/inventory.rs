//! Assets manager.
use glob::glob;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fs::read_to_string;
use std::path::PathBuf;
use yaml_rust2::{Yaml, YamlLoader};

use crate::{EKOError, Result};

/// Headers are in yaml files.
const HEADER_EXT: &str = "*.yaml";

/// Header type in an inventory.
pub(crate) trait HeaderT {
    /// Load from yaml.
    fn load_from_yaml(yml: &Yaml) -> Result<Self>
    where
        Self: Sized;
    /// Comparator.
    fn eq(&self, other: &Self, ulps: i64) -> bool;
}

/// Value type in an inventory.
pub(crate) trait ValueT {
    // File suffix (instead of header suffix)
    const FILE_SUFFIX: &'static str;
    /// Load from file.
    fn load_from_path(&mut self, p: PathBuf) -> Result<()>;
}

/// Assets manager.
pub(crate) struct Inventory<K: HeaderT> {
    /// Working directory
    pub(crate) path: PathBuf,
    /// Available keys
    pub(crate) keys: HashMap<OsString, K>,
}

impl<K: HeaderT> Inventory<K> {
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
                K::load_from_yaml(&cnt[0])?,
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
    pub fn has(&self, k: &K, ulps: i64) -> bool {
        self.keys.iter().any(|it| (it.1).eq(k, ulps))
    }

    /// Load `k` from disk.
    pub fn load<V: ValueT>(&self, k: &K, ulps: i64, v: &mut V) -> Result<()> {
        let k = self.keys.iter().find(|it| (it.1).eq(k, ulps));
        let k = k.ok_or(EKOError::KeyError("because it was not found".to_owned()))?;
        let path = self.path.join(k.0).with_extension(V::FILE_SUFFIX);
        v.load_from_path(path)
    }
}
