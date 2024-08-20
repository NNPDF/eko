use assert_fs::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;

use dekoder::{EvolutionPoint, Operator, EKO};

/// Get v0.15 test object.
fn v015tar() -> PathBuf {
    let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();
    let src = base.join("data").join("v0.15.tar");
    assert!(predicate::path::exists().eval(&src));
    src
}

#[test]
fn open() {
    let src = v015tar();
    let dst = assert_fs::TempDir::new().unwrap();
    // open
    let eko = EKO::read(src.to_owned(), dst.to_owned()).unwrap();
    let metadata = dst.child("metadata.yaml");
    metadata.assert(predicate::path::exists());
    eko.close().unwrap();
}

#[test]
fn close() {
    let src = v015tar();
    let dst = assert_fs::TempDir::new().unwrap();
    {
        // open + close
        let eko = EKO::read(src.to_owned(), dst.to_owned()).unwrap();
        eko.close().unwrap();
    }
    dst.assert(predicate::path::missing());
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
    eko.overwrite_and_close().unwrap();
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
    assert!(ep.eq(eko.available_operators()[0]));
    assert!(eko.has_operator(&ep));
    eko.close().unwrap();
}

#[test]
fn load_operator() {
    let src = v015tar();
    let dst = assert_fs::TempDir::new().unwrap();
    // open
    let eko = EKO::read(src.to_owned(), dst.to_owned()).unwrap();
    // load
    let ep = EvolutionPoint {
        scale: 10000.,
        nf: 4,
    };
    let mut op = Operator::new();
    eko.load_operator(&ep, &mut op).unwrap();
    assert!(op.op.is_some());
    assert!(op.op.unwrap().dim().0 > 0);
    eko.close().unwrap();
}
