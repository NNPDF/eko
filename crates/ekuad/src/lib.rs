use ekore;

#[no_mangle]
pub unsafe extern "C" fn quad_ker(x: f64, extra: bool) -> f64 {
    ekore::ciao(x * 2.0, 3.0)
}
