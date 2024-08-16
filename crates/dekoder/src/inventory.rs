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
    fn load_from_path(p: PathBuf) -> Self;
}

/// Assets manager.
pub(crate) struct Inventory<K: HeaderT, V: ValueT> {
    /// Working directory
    pub(crate) path: PathBuf,
    /// Available keys
    pub(crate) keys: HashMap<OsString, K>,
    /// Available values (in memory)
    pub(crate) values: HashMap<OsString, V>,
}

impl<K: HeaderT, V: ValueT> Inventory<K, V> {
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
    /// Check if `k` is available (with given precision).
    pub fn get(&mut self, k: &K, ulps: i64) -> Option<V> {
        let k = self.keys.iter().find(|it| (it.1).eq(&k, ulps));
        let k = k?;
        let path = self.path.join(k.0).with_extension(V::FILE_SUFFIX);
        let v = V::load_from_path(path);
        // self.values.insert(k.0.clone(), &v);
        Some(v)
    }
}

// mod test {
//     #[test]
//     fn save_as_other() {
//     }
// }
