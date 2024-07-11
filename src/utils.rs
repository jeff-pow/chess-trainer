use std::{fs::File, io::Write};

#[allow(dead_code)]
pub fn write_to_bin<T>(to_bin: &T, file_name: &str, pad: bool) {
    let mut out = File::create(file_name).unwrap();
    let size = size_of::<T>();
    let buf: &[u8] =
        unsafe { core::slice::from_raw_parts((to_bin as *const T) as *const u8, size) };
    let _ = out.write_all(buf);

    if pad {
        let padding = vec![0u8; (64 - size % 64) % 64];
        out.write_all(&padding).unwrap();
    }
}
