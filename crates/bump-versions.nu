cd ..

let version = "0.1.1-alpha.1"

open Cargo.toml | update workspace.package.version $version
                | collect { save -f Cargo.toml }

let crates = open crates/release.json

def update-manifest [] {
  let manifest = $in
  let internals = $manifest | get dependencies | columns | where {|it| $it in $crates}
  $internals | reduce --fold $manifest {|i, acc|
      let field = $"dependencies.($i).version" | split row '.' | into cell-path
      $acc | update $field $version
    }
}

def replace-manifest [] {
  let path = $"crates/($in)/Cargo.toml"
  open $path | update-manifest | to toml | collect { save -f $path }
  $path
}

$crates | each {|p| $p | replace-manifest} | prepend Cargo.toml
