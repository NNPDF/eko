//! Build script to compile the mock `libome` C++ library from `extras/gsoc/libome`
//! and link it statically into `ekore`.

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let libome_dir = manifest_dir.join("../../extras/gsoc/libome");

    println!(
        "cargo:rerun-if-changed={}",
        libome_dir.join("ome.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        libome_dir.join("ome.h").display()
    );

    cc::Build::new()
        .cpp(true)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("/std:c++14")
        .include(&libome_dir)
        .file(libome_dir.join("ome.cpp"))
        .compile("ome");
}
