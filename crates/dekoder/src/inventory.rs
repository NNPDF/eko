//! Assets manager.
use glob::glob;
use hashbrown::HashMap;
use std::ffi::OsString;
use std::fs::read_to_string;
use std::path::PathBuf;
use yaml_rust2::{Yaml, YamlLoader};

/// Headers are in yaml files.
const HEADER_EXT: &'static str = "*.yaml";

/// Header type in an inventory.
pub(crate) trait HeaderT {
    /// Load from yaml.
    fn load_from_yaml(yml: &Yaml) -> Self;
    /// Comparator.
    fn eq(&self, other: &Self, ulps: i64) -> bool;
}

/// Value type in an inventory.
pub(crate) trait ValueT {
    // File suffix (instead of header suffix)
    const FILE_SUFFIX: &'static str;
    /// Load from file.
    fn load_from_path(&mut self, p: PathBuf);
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
    pub fn load_keys(&mut self) {
        for entry in glob(self.path.join(&HEADER_EXT).to_str().unwrap())
            .expect("Failed to read glob pattern")
            .filter(|x| x.is_ok())
            .map(|x| x.unwrap())
        {
            let cnt = YamlLoader::load_from_str(&read_to_string(&entry).unwrap()).unwrap();
            self.keys.insert(
                entry.file_name().unwrap().to_os_string(),
                K::load_from_yaml(&cnt[0]),
            );
        }
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
        self.keys.iter().find(|it| (it.1).eq(&k, ulps)).is_some()
    }

    /// Load `k` from disk.
    pub fn load<V: ValueT>(&mut self, k: &K, ulps: i64, v: &mut V) {
        let k = self.keys.iter().find(|it| (it.1).eq(&k, ulps));
        let k = k.unwrap();
        let path = self.path.join(k.0).with_extension(V::FILE_SUFFIX);
        v.load_from_path(path);
    }
}

// mod test {
//     #[test]
//     fn save_as_other() {
//     }
// }
