//! |N3LO| |QCD|.

// pub mod ggg;
// pub mod ggq;
// pub mod gqg;
pub mod gps;

pub mod gnsm;
pub mod gnsp;
pub mod gnsv;

// /// Compute the singlet anomalous dimension matrix.
// pub fn gamma_singlet(c: &mut Cache, nf: u8, variation: [u8; 4]) -> [[Complex<f64>; 2]; 2] {
//     let gamma_qq = gamma_nsp(c, nf, variation[3]) + gamma_ps(c, nf, variation[3]);
//     [
//         [gamma_qq, gamma_qg(c, nf, variation[2])],
//         [gamma_gq(c, nf, variation[1]), gamma_gg(c, nf, variation[0])],
//     ]
// }

#[cfg(test)]
mod tests {}
