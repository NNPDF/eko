//! Shared bodies for the `#[pyfunction]`s defined throughout this crate.

/// Body for a non-singlet tower function returning shape `(order_qcd,)`.
macro_rules! gamma_ns_qcd_body {
    // With an `n3lo_variation` argument.
    (
        $py:expr, $order_qcd:expr, $mode:expr, $cache:expr, $nf:expr, $n3lo_variation:expr,
        $bound:expr, $path:path
    ) => {{
        gamma_ns_qcd_body!(@check $order_qcd, $mode, $bound);
        let mut cache = $cache.borrow_mut();
        let gamma = $path($order_qcd, $mode, &mut cache.inner, $nf, $n3lo_variation);
        gamma_ns_qcd_body!(@collect $py, $order_qcd, gamma)
    }};
    // Without an `n3lo_variation` argument.
    (
        $py:expr, $order_qcd:expr, $mode:expr, $cache:expr, $nf:expr,
        $bound:expr, $path:path
    ) => {{
        gamma_ns_qcd_body!(@check $order_qcd, $mode, $bound);
        let mut cache = $cache.borrow_mut();
        let gamma = $path($order_qcd, $mode, &mut cache.inner, $nf);
        gamma_ns_qcd_body!(@collect $py, $order_qcd, gamma)
    }};
    (@check $order_qcd:expr, $mode:expr, $bound:expr) => {
        if $bound {
            return Err(PyValueError::new_err(format!(
                "order_qcd out of the supported range, got {}",
                $order_qcd
            )));
        }
        if !matches!($mode, $crate::constants::PID_NSP | $crate::constants::PID_NSM | $crate::constants::PID_NSV) {
            return Err(PyValueError::new_err(format!(
                "invalid non-singlet mode: {}",
                $mode
            )));
        }
    };
    (@collect $py:expr, $order_qcd:expr, $gamma:expr) => {{
        let data: Vec<Complex64> = $gamma
            .into_iter()
            .take($order_qcd)
            .map(|c| Complex64::new(c.re, c.im))
            .collect();
        Ok(PyArray1::from_vec($py, data))
    }};
}

/// Body for a singlet/valence |QCD| matrix tower function returning shape `(order_qcd, dim, dim)`.
macro_rules! gamma_singlet_qcd_body {
    // With an `n3lo_variation` argument.
    (
        $py:expr, $order_qcd:expr, $cache:expr, $nf:expr, $n3lo_variation:expr,
        $bound:expr, $path:path, $dim:expr
    ) => {{
        gamma_singlet_qcd_body!(@check $order_qcd, $bound);
        let mut cache = $cache.borrow_mut();
        let gamma = $path($order_qcd, &mut cache.inner, $nf, $n3lo_variation);
        gamma_singlet_qcd_body!(@collect $py, $order_qcd, gamma, $dim)
    }};
    // Without an `n3lo_variation` argument.
    (
        $py:expr, $order_qcd:expr, $cache:expr, $nf:expr,
        $bound:expr, $path:path, $dim:expr
    ) => {{
        gamma_singlet_qcd_body!(@check $order_qcd, $bound);
        let mut cache = $cache.borrow_mut();
        let gamma = $path($order_qcd, &mut cache.inner, $nf);
        gamma_singlet_qcd_body!(@collect $py, $order_qcd, gamma, $dim)
    }};
    (@check $order_qcd:expr, $bound:expr) => {
        if $bound {
            return Err(PyValueError::new_err(format!(
                "order_qcd out of the supported range, got {}",
                $order_qcd
            )));
        }
    };
    (@collect $py:expr, $order_qcd:expr, $gamma:expr, $dim:expr) => {{
        let mut data: Vec<Complex64> = Vec::with_capacity($order_qcd * $dim * $dim);
        for mat in $gamma.into_iter().take($order_qcd) {
            for row in mat.iter() {
                for v in row.iter() {
                    data.push(Complex64::new(v.re, v.im));
                }
            }
        }
        PyArray1::from_vec($py, data)
            .reshape([$order_qcd, $dim, $dim])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }};
}

/// Body for a |QCD| x |QED| non-singlet tower function returning shape
/// `(order_qcd + 1, order_qed + 1)`.
macro_rules! gamma_ns_qed_body {
    (
        $py:expr, $order_qcd:expr, $order_qed:expr, $mode:expr, $cache:expr, $nf:expr,
        $n3lo_variation:expr, $bound:expr, $path:path
    ) => {{
        if $bound {
            return Err(PyValueError::new_err(format!(
                "order_qcd/order_qed out of the supported range, got {}, {}",
                $order_qcd, $order_qed
            )));
        }
        if !matches!(
            $mode,
            $crate::constants::PID_NSP_U
                | $crate::constants::PID_NSP_D
                | $crate::constants::PID_NSM_U
                | $crate::constants::PID_NSM_D
                | $crate::constants::PID_NSP
                | $crate::constants::PID_NSM
                | $crate::constants::PID_NSV,
        ) {
            return Err(PyValueError::new_err(format!(
                "invalid non-singlet mode: {}",
                $mode
            )));
        }

        let mut cache = $cache.borrow_mut();
        let gamma = $path(
            $order_qcd,
            $order_qed,
            $mode,
            &mut cache.inner,
            $nf,
            $n3lo_variation,
        );

        let mut data: Vec<Complex64> = Vec::with_capacity(($order_qcd + 1) * ($order_qed + 1));
        for row in gamma.into_iter().take($order_qcd + 1) {
            for v in row.into_iter().take($order_qed + 1) {
                data.push(Complex64::new(v.re, v.im));
            }
        }

        PyArray1::from_vec($py, data)
            .reshape([$order_qcd + 1, $order_qed + 1])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }};
}

/// Body for a |QCD| x |QED| matrix tower function (singlet/valence) returning shape
/// `(order_qcd + 1, order_qed + 1, dim, dim)`.
macro_rules! gamma_qed_matrix_body {
    (
        $py:expr, $order_qcd:expr, $order_qed:expr, $cache:expr, $nf:expr, $n3lo_variation:expr,
        $bound:expr, $path:path, $dim:expr
    ) => {{
        if $bound {
            return Err(PyValueError::new_err(format!(
                "order_qcd/order_qed out of the supported range, got {}, {}",
                $order_qcd, $order_qed
            )));
        }

        let mut cache = $cache.borrow_mut();
        let gamma = $path(
            $order_qcd,
            $order_qed,
            &mut cache.inner,
            $nf,
            $n3lo_variation,
        );

        let mut data: Vec<Complex64> =
            Vec::with_capacity(($order_qcd + 1) * ($order_qed + 1) * $dim * $dim);
        for row in gamma.into_iter().take($order_qcd + 1) {
            for mat in row.into_iter().take($order_qed + 1) {
                for r in mat.iter() {
                    for v in r.iter() {
                        data.push(Complex64::new(v.re, v.im));
                    }
                }
            }
        }

        PyArray1::from_vec($py, data)
            .reshape([$order_qcd + 1, $order_qed + 1, $dim, $dim])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }};
}

/// Body for an |OME| matrix tower function returning shape `(matching_order_qcd, dim, dim)`.
macro_rules! ome_matrix_body {
    (
        $py:expr, $order:expr, $cache:expr, $nf:expr, $l:expr,
        $bound:expr, $path:path, $dim:expr
    ) => {{
        if $bound {
            return Err(PyValueError::new_err(format!(
                "matching_order_qcd out of the supported range, got {}",
                $order
            )));
        }

        let mut cache = $cache.borrow_mut();
        let ome = $path($order, &mut cache.inner, $nf, $l);

        let mut data: Vec<Complex64> = Vec::with_capacity($order * $dim * $dim);
        for mat in ome.into_iter().take($order) {
            for row in mat.iter() {
                for v in row.iter() {
                    data.push(Complex64::new(v.re, v.im));
                }
            }
        }

        PyArray1::from_vec($py, data)
            .reshape([$order, $dim, $dim])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }};
}
