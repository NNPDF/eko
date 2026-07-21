//! Shared bodies for the C API functions defined throughout this crate.

/// Body of a `*_result_len` function: bounds-check then compute the buffer size.
macro_rules! result_len_body {
    ($bound:expr, $formula:expr) => {{
        if $bound {
            return 0;
        }
        $formula
    }};
}

/// Body of a non-singlet QCD anomalous dimension tower function.
macro_rules! gamma_ns_qcd_body {
    (
        $order_qcd:expr, $mode:expr, $cache:expr, $nf:expr, $n3lo_variation:expr,
        $result:expr, $bound:expr, $path:path
    ) => {
        gamma_ns_qcd_body!(@inner
            $order_qcd, $mode, $cache, $result, $bound,
            {
                if $n3lo_variation.is_null() { return; }
                let var: [u8; 3] = std::slice::from_raw_parts($n3lo_variation, 3).try_into().unwrap();
                $path($order_qcd, $mode, &mut *$cache, $nf, var)
            }
        )
    };
    (
        $order_qcd:expr, $mode:expr, $cache:expr, $nf:expr,
        $result:expr, $bound:expr, $path:path
    ) => {
        gamma_ns_qcd_body!(@inner
            $order_qcd, $mode, $cache, $result, $bound,
            { $path($order_qcd, $mode, &mut *$cache, $nf) }
        )
    };
    (@inner $order_qcd:expr, $mode:expr, $cache:expr, $result:expr, $bound:expr, $eval_block:block) => {{
        if $cache.is_null() || $result.is_null() || $bound {
            return;
        }
        if !matches!($mode, $crate::PID_NSP | $crate::PID_NSM | $crate::PID_NSV) {
            return;
        }
        unsafe {
            let out = std::slice::from_raw_parts_mut($result, $order_qcd);
            for (dst, src) in out.iter_mut().zip($eval_block) {
                *dst = src.into();
            }
        }
    }};
}

/// Body of a singlet QCD anomalous dimension matrix tower function.
macro_rules! gamma_singlet_qcd_body {
    (
        $order_qcd:expr, $cache:expr, $nf:expr, $n3lo_variation:expr,
        $result:expr, $bound:expr, $path:path
    ) => {
        gamma_singlet_qcd_body!(@inner
            $order_qcd, $cache, $result, $bound,
            {
                if $n3lo_variation.is_null() { return; }
                let var: [u8; 4] = std::slice::from_raw_parts($n3lo_variation, 4).try_into().unwrap();
                $path($order_qcd, &mut *$cache, $nf, var)
            }
        )
    };
    (
        $order_qcd:expr, $cache:expr, $nf:expr,
        $result:expr, $bound:expr, $path:path
    ) => {
        gamma_singlet_qcd_body!(@inner
            $order_qcd, $cache, $result, $bound,
            { $path($order_qcd, &mut *$cache, $nf) }
        )
    };
    (@inner $order_qcd:expr, $cache:expr, $result:expr, $bound:expr, $eval_block:block) => {{
        if $cache.is_null() || $result.is_null() || $bound {
            return;
        }
        unsafe {
            let out = std::slice::from_raw_parts_mut($result, $order_qcd * 4);
            for (o, mat) in $eval_block.iter().take($order_qcd).enumerate() {
                for r in 0..2_usize {
                    for col in 0..2_usize {
                        out[o * 4 + r * 2 + col] = mat[r][col].into();
                    }
                }
            }
        }
    }};
}

/// Body of a non-singlet |QCD| x |QED| anomalous dimension tower function.
macro_rules! gamma_ns_qed_body {
    (
        $order_qcd:expr, $order_qed:expr, $mode:expr, $cache:expr, $nf:expr,
        $n3lo_variation:expr, $result:expr, $bound:expr, $path:path
    ) => {{
        // Sanity checks
        if $cache.is_null() || $n3lo_variation.is_null() || $result.is_null() {
            return;
        }
        if $bound {
            return;
        }
        if !matches!(
            $mode,
            $crate::PID_NSP_U
                | $crate::PID_NSP_D
                | $crate::PID_NSM_U
                | $crate::PID_NSM_D
                | $crate::PID_NSP
                | $crate::PID_NSM
                | $crate::PID_NSV
        ) {
            return;
        }
        unsafe {
            let cache = &mut *$cache;
            let var: [u8; 3] = std::slice::from_raw_parts($n3lo_variation, 3)
                .try_into()
                .unwrap();
            let ncols = $order_qed + 1;
            let out = std::slice::from_raw_parts_mut($result, ($order_qcd + 1) * ncols);
            // Call function and transfer result
            let gamma = $path($order_qcd, $order_qed, $mode, cache, $nf, var);
            for (i, row) in gamma.iter().take($order_qcd + 1).enumerate() {
                for (j, val) in row.iter().take(ncols).enumerate() {
                    out[i * ncols + j] = (*val).into();
                }
            }
        }
    }};
}

/// Body of a |QCD| x |QED| anomalous dimension matrix tower function.
macro_rules! gamma_qed_matrix_body {
    (
        $order_qcd:expr, $order_qed:expr, $cache:expr, $nf:expr, $n3lo_variation:expr,
        $result:expr, $bound:expr, $path:path, $dim:expr, $elem:expr, $n3lo_size:expr
    ) => {{
        // Sanity checks
        if $cache.is_null() || $n3lo_variation.is_null() || $result.is_null() {
            return;
        }
        if $bound {
            return;
        }
        unsafe {
            let cache = &mut *$cache;
            let var: [u8; $n3lo_size] = std::slice::from_raw_parts($n3lo_variation, $n3lo_size)
                .try_into()
                .unwrap();
            let ncols = $order_qed + 1;
            let out = std::slice::from_raw_parts_mut($result, ($order_qcd + 1) * ncols * $elem);
            // Call function and transfer result
            let gamma = $path($order_qcd, $order_qed, cache, $nf, var);
            for (i, row) in gamma.iter().take($order_qcd + 1).enumerate() {
                for (j, mat) in row.iter().take(ncols).enumerate() {
                    let base = (i * ncols + j) * $elem;
                    for r in 0..$dim {
                        for col in 0..$dim {
                            out[base + r * $dim + col] = mat[r][col].into();
                        }
                    }
                }
            }
        }
    }};
}

/// Body of an OME matrix tower function.
macro_rules! ome_matrix_body {
    (
        $order:expr, $cache:expr, $nf:expr, $l:expr, $result:expr,
        $bound:expr, $path:path, $dim:expr, $elem:expr
    ) => {{
        // Sanity checks
        if $cache.is_null() || $result.is_null() {
            return;
        }
        if $bound {
            return;
        }
        unsafe {
            let cache = &mut *$cache;
            let out = std::slice::from_raw_parts_mut($result, $order * $elem);
            // Call function and transfer result
            for (o, mat) in $path($order, cache, $nf, $l)
                .iter()
                .take($order)
                .enumerate()
            {
                for r in 0..$dim {
                    for col in 0..$dim {
                        out[o * $elem + r * $dim + col] = mat[r][col].into();
                    }
                }
            }
        }
    }};
}
