//! Define EKO metadata
use serde::Deserialize;

type FlavorsNumber = u8;
type Scale = f64;
type X = f64;
type Pid = i32;

#[derive(Deserialize)]
struct EvolutionPoint(Scale, FlavorsNumber);

/// Manipulation information, describing the current status of the EKO (e.g. `input_grid` and
/// `target_grid`).
#[derive(Deserialize)]
struct Bases {
    x_grid: Vec<X>,
    pids: Vec<Pid>,
    #[serde(alias = "_inputgrid")]
    input_grid: Option<Vec<X>>,
    #[serde(alias = "_inputpids")]
    input_pids: Option<Vec<Pid>>,
    #[serde(alias = "_targetgrid")]
    target_grid: Option<Vec<X>>,
    #[serde(alias = "_targetpids")]
    target_pids: Option<Vec<Pid>>,
}

#[derive(Deserialize)]
struct Metadata {
    origin: EvolutionPoint,
    bases: Bases,
}
