use assert_fs::prelude::*;
use predicates::prelude::*;
use std::path::PathBuf;

use dekoder::eko::{EvolutionPoint, EKO};

// assert_fs will clean up the directories for us,
// so for the most part we don't need worry about that.

/// Get v0.15 test object.
fn v3tar() -> PathBuf {
    let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();
    let src = base.join("data").join("v3.tar");
    assert!(predicate::path::exists().eval(&src));
    src
}

#[test]
fn open() {
    let src = v3tar();
    let dst = assert_fs::TempDir::new().unwrap();
    // open
    let _eko = EKO::extract(src.to_owned(), dst.to_owned()).unwrap();
    let metadata = dst.child("metadata.yaml");
    metadata.assert(predicate::path::exists());
}

#[test]
fn destroy() {
    let src = v3tar();
    let dst = assert_fs::TempDir::new().unwrap();
    {
        // extract + destroy
        let eko = EKO::extract(src.to_owned(), dst.to_owned()).unwrap();
        eko.destroy().unwrap();
    }
    dst.assert(predicate::path::missing());
}

#[test]
fn save_as_other() {
    let src = v3tar();
    let dst = assert_fs::TempDir::new().unwrap();
    // open
    let eko = EKO::extract(src.to_owned(), dst.to_owned()).unwrap();
    // write to somewhere else
    let tarb = assert_fs::NamedTempFile::new("write_test.tar").unwrap();
    eko.write(tarb.to_owned()).unwrap();
    tarb.assert(predicate::path::exists());
}

#[test]
fn has_operator() {
    let src = v3tar();
    let dst = assert_fs::TempDir::new().unwrap();
    // open
    let eko = EKO::extract(src.to_owned(), dst.to_owned()).unwrap();
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
}

#[test]
fn load_operator() {
    let src = v3tar();
    let dst = assert_fs::TempDir::new().unwrap();
    // open
    let eko = EKO::extract(src.to_owned(), dst.to_owned()).unwrap();
    // load
    let ep = EvolutionPoint {
        scale: 10000.,
        nf: 4,
    };
    let operator = eko.load_operator(&ep).unwrap();
    assert!(operator.op.is_some());
    assert!(operator.err.is_some());
    let op = operator.op.unwrap();
    assert!(op.dim().0 > 0);
    assert!(op.dim().0 == operator.err.unwrap().dim().0);
}
