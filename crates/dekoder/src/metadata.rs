use serde::Deserialize;

type FlavorsNumber = u8;
type Scale = f64;
type X = f64;
type Pid = i32;
struct EvolutionPoint(Scale, FlavorsNumber);

/// Manipulation information, describing the current status of the EKO (e.g. `input_grid` and
/// `target_grid`).
#[derive(Deserialize)]
struct Bases {
    xgrid: Vec<X>,
    pids: Vec<Pid>,
    #[serde(alias = "_inputgrid")]
    input_grid: Option<Vec<X>>,
    #[serde(alias = "_inputpids", with = "either::serde_untagged_optional")]
    input_pids: Option<Either<Vec<Vec<Pid>>, Vec<Pid>>>,
    #[serde(alias = "_targetgrid")]
    target_grid: Option<Vec<X>>,
    #[serde(alias = "_targetpids")]
    target_pids: Option<Vec<Pid>>,
}

#[derive(Deserialize)]
struct Metadata {
    origin: EvolutionPoint,
    inputgrid: Vec<X>,
    inputpids: Vec<Pid>,
    targetgrid: Vec<X>,
    targetpids: Vec<Pid>,
}
