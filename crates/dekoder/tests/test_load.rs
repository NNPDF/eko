use assert_fs::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;

use dekoder::{EvolutionPoint, Operator, EKO};

/// Get v0.15 test object
fn v015tar() -> PathBuf {
    let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();
    let src = base.join("data").join("v0.15.tar");
    assert!(predicate::path::exists().eval(&src));
    src
}

#[test]
fn save_as_other() {
    let src = v015tar();
    let dst = assert_fs::TempDir::new().unwrap();
    // open
    let mut eko = EKO::edit(src.to_owned(), dst.to_owned()).unwrap();
    // write to somewhere else
    let tarb = assert_fs::NamedTempFile::new("v0.15b.tar").unwrap();
    eko.set_tar_path(tarb.to_owned());
    eko.close(true).unwrap();
    tarb.assert(predicate::path::exists());
}

#[test]
fn has_operator() {
    let src = v015tar();
    let dst = assert_fs::TempDir::new().unwrap();
    // open
    let eko = EKO::read(src.to_owned(), dst.to_owned()).unwrap();
    // check there is only one:
    assert!(eko.available_operators().len() == 1);
    // ask for one
    let ep = EvolutionPoint {
        scale: 10000.,
        nf: 4,
    };
    // it is the one
    assert!(ep.equals(eko.available_operators()[0], 64));
    assert!(eko.has_operator(&ep, 64));
    eko.close(false).unwrap();
}

#[test]
fn load_operator() {
    let src = v015tar();
    let dst = assert_fs::TempDir::new().unwrap();
    // open
    let mut eko = EKO::read(src.to_owned(), dst.to_owned()).unwrap();
    // load
    let ep = EvolutionPoint {
        scale: 10000.,
        nf: 4,
    };
    let mut op = Operator::zeros();
    eko.load_operator(&ep, 64, &mut op).unwrap();
    assert!(op.op.dim().0 > 0);
    eko.close(false).unwrap();
}
