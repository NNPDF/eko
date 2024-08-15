//! eko output reader.
use glob::glob;
use hashbrown::HashMap;
use std::fs::remove_dir_all;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

const HEADER_EXT: &'static str = "*.yaml";
struct Inventory {
    /// Working directory
    path: PathBuf,
    /// Available items
    keys: HashMap<String, String>,
}
impl Inventory {
    /// Load all available entries
    pub fn load_keys(&mut self) {
        for entry in glob(self.path.join(HEADER_EXT.to_owned()).to_str().unwrap())
            .expect("Failed to read glob pattern")
            .filter(|x| x.is_ok())
        {
            self.keys.insert(
                (*entry.unwrap().file_name().unwrap())
                    .to_os_string()
                    .into_string()
                    .unwrap(),
                "Blub".to_string(),
            );
        }
    }

    pub fn has_key(&self, ep: String, ulps: f64) -> bool {
        self.keys.values().find(|v| (*v).eq(&ep)).is_some()
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
    operators: Inventory,
}

const DIR_OPERATORS: &'static str = "operators";

impl EKO {
    /// Check our working directory is safe.
    fn check(&self) -> Result<(), std::io::Error> {
        let path_exists = self.path.try_exists().is_ok_and(|x| x);
        if !path_exists {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Working directory not found!",
            ));
        }
        Ok(())
    }

    /// Remove the working directory.
    fn destroy(&self) -> Result<(), std::io::Error> {
        self.check()?;
        remove_dir_all(self.path.to_owned())
    }

    /// Write content back to an archive and destroy working directory.
    pub fn close(&self, allow_overwrite: bool) -> Result<(), std::io::Error> {
        self.write(allow_overwrite, true)
    }

    /// Write content back to an archive.
    pub fn write(&self, allow_overwrite: bool, destroy: bool) -> Result<(), std::io::Error> {
        self.check()?;
        // in read-only there is nothing to do then to destroy, since we couldn't
        if self.read_only && destroy {
            return self.destroy();
        }
        // check we can write
        if self.tar_path.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No target path!",
            ));
        }
        let dst = self.tar_path.to_owned().unwrap();
        let dst_exists = dst.try_exists().is_ok_and(|x| x);
        if !allow_overwrite && dst_exists {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                format!("Target already exists '{}'!", dst.display()),
            ));
        }
        // create writer
        let dst_file = match File::create(dst.to_owned()) {
            Err(why) => panic!("couldn't open {}: {}", dst.display(), why.to_string()),
            Ok(file) => file,
        };
        print!("dst: {}", dst.display());
        let dst_file = BufWriter::with_capacity(128 * 1024, dst_file);
        let mut ar = tar::Builder::new(dst_file);
        // do it!
        ar.append_dir_all(".", self.path.to_owned()).unwrap();
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
    pub fn read(src: PathBuf, dst: PathBuf) -> Self {
        Self::extract(src, dst, true)
    }

    /// Open tar from `src` to `dst` for editing.
    pub fn edit(src: PathBuf, dst: PathBuf) -> Self {
        Self::extract(src, dst, false)
    }

    /// Extract tar file from `src` to `dst`.
    pub fn extract(src: PathBuf, dst: PathBuf, read_only: bool) -> Self {
        let mut ar = tar::Archive::new(File::open(src.to_owned()).unwrap());
        ar.unpack(dst.to_owned()).unwrap();
        let mut obj = Self::load_opened(dst, read_only);
        obj.set_tar_path(src);
        obj
    }

    /// Load an EKO from a directory `path` (instead of tar).
    pub fn load_opened(path: PathBuf, read_only: bool) -> Self {
        let mut operators = Inventory {
            path: path.join(DIR_OPERATORS),
            keys: HashMap::new(),
        };
        operators.load_keys();
        Self {
            path,
            tar_path: None,
            read_only,
            operators,
        }
    }

    /// Check if the operator at the evolution point `ep` is available.
    pub fn has_operator(&self, ep: String, ulps: f64) -> bool {
        self.operators.has_key(ep, ulps)
    }
}

mod test {
    #[test]
    fn save_as_other() {
        use super::EKO;
        use std::fs::{remove_dir_all, remove_file};
        use std::path::PathBuf;
        let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();
        let src = base.join("data").join("v0.15.tar");
        assert!(src.try_exists().is_ok_and(|x| x));
        let dst = base.join("target").join("v0.15");
        // get rid of previous runs if needed
        let dst_exists = dst.try_exists().is_ok_and(|x| x);
        if dst_exists {
            let _ = remove_dir_all(dst.to_owned());
        }
        // open
        let mut eko = EKO::edit(src.to_owned(), dst.to_owned());
        let dst_exists = dst.try_exists().is_ok_and(|x| x);
        assert!(dst_exists);
        // set a different output
        let tarb = base.join("target").join("v0.15b.tar");
        let tarb_exists = tarb.try_exists().is_ok_and(|x| x);
        if tarb_exists {
            let _ = remove_file(tarb.to_owned());
        }
        eko.set_tar_path(tarb.to_owned());
        // close
        let res = eko.close(true);
        assert!(res.is_ok());
        assert!(tarb.try_exists().is_ok_and(|x| x));
        let dst_exists = dst.try_exists().is_ok_and(|x| x);
        assert!(!dst_exists);
        // clean up
        if dst_exists {
            let _ = remove_dir_all(dst.to_str().unwrap());
        }
    }

    #[test]
    fn read_keys() {
        use super::EKO;
        use std::fs::remove_dir_all;
        use std::path::PathBuf;
        let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();
        let src = base.join("data").join("v0.15.tar");
        assert!(src.try_exists().is_ok_and(|x| x));
        let dst = base.join("target").join("v0.15");
        // get rid of previous runs if needed
        let dst_exists = dst.try_exists().is_ok_and(|x| x);
        if dst_exists {
            let _ = remove_dir_all(dst.to_owned());
        }
        // open
        let eko = EKO::read(src.to_owned(), dst.to_owned());
        assert!(eko.has_operator("Blub".to_string(), 1e-7));
    }
}
