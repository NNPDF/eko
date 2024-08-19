use dekoder::{EvolutionPoint, Operator, EKO};
use std::fs::{remove_dir_all, remove_file};
use std::path::PathBuf;

#[test]
fn save_as_other() {
    let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();
    let src = base.join("data").join("v0.15.tar");
    assert!(src.try_exists().is_ok_and(|x| x));
    let dst = base.join("target").join("v0.15a");
    // get rid of previous runs if needed
    let dst_exists = dst.try_exists().is_ok_and(|x| x);
    if dst_exists {
        remove_dir_all(dst.to_owned()).unwrap();
    }
    // open
    let mut eko = EKO::edit(src.to_owned(), dst.to_owned());
    let dst_exists = dst.try_exists().is_ok_and(|x| x);
    assert!(dst_exists);
    // set a different output
    let tarb = base.join("target").join("v0.15b.tar");
    let tarb_exists = tarb.try_exists().is_ok_and(|x| x);
    if tarb_exists {
        remove_file(tarb.to_owned()).unwrap();
    }
    eko.set_tar_path(tarb.to_owned());
    // close
    eko.close(true).unwrap();
    let tarb_exists = tarb.try_exists().is_ok_and(|x| x);
    assert!(tarb_exists);
    let dst_exists = dst.try_exists().is_ok_and(|x| x);
    assert!(!dst_exists);
    // clean up
    if tarb_exists {
        remove_file(tarb.to_owned()).unwrap();
    }
    if dst_exists {
        remove_dir_all(dst.to_str().unwrap()).unwrap();
    }
}

#[test]
fn has_operator() {
    let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();
    let src = base.join("data").join("v0.15.tar");
    assert!(src.try_exists().is_ok_and(|x| x));
    let dst = base.join("target").join("v0.15b");
    // get rid of previous runs if needed
    let dst_exists = dst.try_exists().is_ok_and(|x| x);
    if dst_exists {
        remove_dir_all(dst.to_owned()).unwrap();
    }
    // open
    let eko = EKO::read(src.to_owned(), dst.to_owned());
    // check there is only one:
    assert!(eko.available_operators().len() == 1);
    // ask for one
    let ep = EvolutionPoint {
        scale: 10000.,
        nf: 4,
    };
    assert!(eko.has_operator(&ep, 64));
    eko.close(false).unwrap();
}

#[test]
fn load_operator() {
    let base: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();
    let src = base.join("data").join("v0.15.tar");
    assert!(src.try_exists().is_ok_and(|x| x));
    let dst = base.join("target").join("v0.15c");
    // get rid of previous runs if needed
    let dst_exists = dst.try_exists().is_ok_and(|x| x);
    if dst_exists {
        remove_dir_all(dst.to_owned()).unwrap();
    }
    // open
    let mut eko = EKO::read(src.to_owned(), dst.to_owned());
    let ep = EvolutionPoint {
        scale: 10000.,
        nf: 4,
    };
    let mut op = Operator::zeros();
    eko.load_operator(&ep, 64, &mut op);
    assert!(op.ar.dim().0 > 0);
    eko.close(false).unwrap();
}
