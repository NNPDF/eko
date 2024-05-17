cd ..

let version = "x.x.x"

open Cargo.toml | update workspace.package.version $version | save -f Cargo.toml

let crates = ls crates | where type == dir | get name | filter {|n| $"($n)/Cargo.toml" | path exists } | each {|p| split row "/" | last}

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
  open $path | update-manifest | to toml | save -f $path
  $path
}

$crates | each {|p| $p | replace-manifest} | prepend Cargo.toml
