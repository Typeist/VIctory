//! What a read-back frame holds, as numbers: is it black, or is there a
//! picture? Rows of colour pixels in, a summary and a PPM file out.
//!
//! Nothing here names a GPU or window type, so the arithmetic runs (and is
//! tested) on a host with no display: the D3D12 side hands over the bytes it
//! mapped and takes back a [`Summary`] to log.
#![cfg_attr(not(windows), allow(dead_code))]

use std::path::Path;

/// The pixel layout of a read-back colour target: what the swap chain can
/// hold, which is what a presented frame is made of.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Pixels {
    /// `R8G8B8A8` and its family: bytes are red, green, blue, alpha.
    Rgba8,
    /// `B8G8R8A8` and its family: bytes are blue, green, red, alpha.
    Bgra8,
    /// `R10G10B10A2`: one little-endian `u32` of 10-bit channels.
    Rgb10a2,
}

impl Pixels {
    /// The layout of a `DXGI_FORMAT` value, if it is one of the formats a
    /// frame is presented in that this reads. `None` for everything else
    /// (float targets, single channels, depth): the caller says so once.
    pub fn of(format: u32) -> Option<Pixels> {
        Some(match format {
            // R8G8B8A8_UNORM, _SRGB, the snorm/typed variants.
            27..=32 => Pixels::Rgba8,
            // B8G8R8A8_UNORM, _SRGB, and the X8 variant.
            87 | 90 | 91 => Pixels::Bgra8,
            // R10G10B10A2_UNORM and its family.
            23..=25 => Pixels::Rgb10a2,
            _ => return None,
        })
    }

    /// A short name for log lines.
    pub fn name(self) -> &'static str {
        match self {
            Pixels::Rgba8 => "rgba8",
            Pixels::Bgra8 => "bgra8",
            Pixels::Rgb10a2 => "rgb10a2",
        }
    }

    /// Bytes a pixel takes in a row.
    fn width(self) -> usize {
        4
    }
}

/// One pixel's bytes at `off` in `row`, zero-filled if the row is short.
fn at(row: &[u8], off: usize, n: usize) -> [u8; 4] {
    let mut p = [0u8; 4];
    if let Some(s) = row.get(off..off + n) {
        p[..n].copy_from_slice(s);
    }
    p
}

/// Copies `height` rows of `pixels`, `pitch` bytes apart, as tightly packed
/// 8-bit RGBA (alpha 255). 10-bit channels are scaled to 8 bits.
pub fn pack(pixels: Pixels, width: u32, height: u32, pitch: usize, rows: &[u8]) -> Vec<u8> {
    let (w, h) = (width as usize, height as usize);
    let n = pixels.width();
    let mut out = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        let row = rows.get(y * pitch..).unwrap_or(&[]);
        for x in 0..w {
            match pixels {
                Pixels::Rgba8 => {
                    let p = at(row, x * n, 3);
                    out.extend_from_slice(&[p[0], p[1], p[2], 255]);
                }
                Pixels::Bgra8 => {
                    let p = at(row, x * n, 3);
                    out.extend_from_slice(&[p[2], p[1], p[0], 255]);
                }
                Pixels::Rgb10a2 => {
                    let p = at(row, x * n, 4);
                    let v = u32::from_le_bytes(p);
                    let channel = |shift: u32| (((v >> shift) & 0x3ff) * 255 / 1023) as u8;
                    out.extend_from_slice(&[channel(0), channel(10), channel(20), 255]);
                }
            }
        }
    }
    out
}

/// What a frame's pixels hold: enough to tell a black frame from a drawn
/// one without opening the image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub width: u32,
    pub height: u32,
    /// Pixels read (a frame with none is black and broken).
    pub pixels: u64,
    /// Pixels with any channel above zero.
    pub nonzero: u64,
    /// Per channel (red, green, blue) over every pixel.
    pub min: [u8; 3],
    pub max: [u8; 3],
    pub mean: [f64; 3],
    /// Distinct 16-wide luminance levels that have at least one pixel
    /// (1 for a flat frame, many for a picture).
    pub buckets: u32,
    /// The highest level with a pixel: 0 for a frame that is black all over.
    pub top_bucket: u8,
    /// Every channel of every pixel below 8: black, by the same threshold
    /// the presented-frame probe uses.
    pub black: bool,
}

/// Levels per 0..=255 luminance value: 16 wide, so 0 is "black".
const LEVEL: u16 = 16;

/// Summarises `rgba` (tightly packed, from [`pack`]) as a frame of
/// `width` by `height` pixels.
pub fn summarize(width: u32, height: u32, rgba: &[u8]) -> Summary {
    let mut min = [255u8; 3];
    let mut max = [0u8; 3];
    let mut sum = [0u64; 3];
    let mut nonzero = 0u64;
    let mut pixels = 0u64;
    let mut present = [false; (256 / LEVEL) as usize];
    let mut top_bucket = 0u8;
    for p in rgba.as_chunks::<4>().0 {
        pixels += 1;
        let c = [p[0], p[1], p[2]];
        if c.iter().any(|&v| v != 0) {
            nonzero += 1;
        }
        for (i, &v) in c.iter().enumerate() {
            min[i] = min[i].min(v);
            max[i] = max[i].max(v);
            sum[i] += u64::from(v);
        }
        // 0.30, 0.59, 0.11 in integers, staying exact.
        let lum = (u16::from(c[0]) * 30 + u16::from(c[1]) * 59 + u16::from(c[2]) * 11) / 100;
        let bucket = (lum / LEVEL) as u8;
        present[bucket as usize] = true;
        top_bucket = top_bucket.max(bucket);
    }
    // An empty frame has no low value to report; do not claim 255.
    if pixels == 0 {
        min = [0; 3];
    }
    let mean = sum.map(|s| s as f64 / pixels.max(1) as f64);
    let brightest = *max.iter().max().unwrap_or(&0);
    Summary {
        width,
        height,
        pixels,
        nonzero,
        min,
        max,
        mean,
        buckets: present.iter().filter(|p| **p).count() as u32,
        top_bucket,
        black: brightest < 8,
    }
}

/// Writes `rgba` (tightly packed, from [`pack`]) as a binary PPM (`P6`, which
/// has no alpha), creating the folder. PPM needs no encoder, so a frame can
/// be looked at with any image viewer.
pub fn write_ppm(path: &Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    out.reserve(width as usize * height as usize * 3);
    for p in rgba.as_chunks::<4>().0 {
        out.extend_from_slice(&p[..3]);
    }
    // A PPM path beside PNGs is expected to end in .ppm; nothing reads it.
    std::fs::write(path, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_of_zeros_is_black() {
        let s = summarize(4, 2, &[0u8; 4 * 8]);
        assert_eq!((s.width, s.height, s.pixels, s.nonzero), (4, 2, 8, 0));
        assert_eq!(s.min, [0, 0, 0]);
        assert_eq!(s.max, [0, 0, 0]);
        assert_eq!(s.mean, [0.0, 0.0, 0.0]);
        assert_eq!(s.buckets, 1, "one level, and it is the black one");
        assert_eq!(s.top_bucket, 0);
        assert!(s.black);
    }

    #[test]
    fn a_drawn_frame_is_not_black() {
        // Two pixels black then two white, in one row.
        let mut rgba = vec![0u8; 4 * 2 * 4];
        for p in rgba.as_chunks_mut::<4>().0.iter_mut().skip(4) {
            p[..3].copy_from_slice(&[255, 255, 255]);
        }
        let s = summarize(4, 2, &rgba);
        assert_eq!(s.nonzero, 4);
        assert_eq!(s.max, [255, 255, 255]);
        assert_eq!(s.min, [0, 0, 0]);
        assert!((s.mean[0] - 127.5).abs() < 1e-9);
        assert_eq!(s.buckets, 2, "black and white, nothing between");
        assert_eq!(s.top_bucket, 15);
        assert!(!s.black);
    }

    #[test]
    fn a_dim_frame_is_black_and_a_bucket_is_sixteen_wide() {
        // 15 is the last value in the black bucket; 16 starts the next one.
        let dim = summarize(1, 1, &[7, 7, 7, 255]);
        assert_eq!(dim.top_bucket, 0);
        assert!(dim.black);
        let lit = summarize(1, 1, &[16, 16, 16, 255]);
        assert_eq!(lit.top_bucket, 1);
        assert!(!lit.black, "16 is above the black threshold of 8");
    }

    #[test]
    fn packing_swaps_blue_and_red_and_reads_ten_bit_channels() {
        // One BGRA pixel: blue 10, green 20, red 30.
        let bgra = pack(Pixels::Bgra8, 1, 1, 4, &[10, 20, 30, 255]);
        assert_eq!(bgra, [30, 20, 10, 255]);
        let rgba = pack(Pixels::Rgba8, 1, 1, 4, &[10, 20, 30, 255]);
        assert_eq!(rgba, [10, 20, 30, 255]);
        // R10G10B10A2: full red, half green, no blue.
        let v: u32 = 1023 | (511 << 10);
        let ten = pack(Pixels::Rgb10a2, 1, 1, 4, &v.to_le_bytes());
        assert_eq!(ten, [255, 127, 0, 255]);
    }

    #[test]
    fn packing_ignores_row_padding() {
        // Two rows, pitch 12 for 2 pixels of 4 bytes: 4 bytes of padding.
        let mut rows = vec![0u8; 24];
        rows[..4].copy_from_slice(&[1, 2, 3, 255]);
        rows[4..8].copy_from_slice(&[4, 5, 6, 255]);
        rows[12..16].copy_from_slice(&[7, 8, 9, 255]);
        rows[16..20].copy_from_slice(&[10, 11, 12, 255]);
        let packed = pack(Pixels::Rgba8, 2, 2, 12, &rows);
        assert_eq!(packed.len(), 16);
        assert_eq!(packed[0..3], [1, 2, 3]);
        assert_eq!(packed[8..11], [7, 8, 9]);
    }

    #[test]
    fn a_ppm_has_the_header_and_three_bytes_a_pixel() {
        let dir = tempfile::tempdir().expect("a temporary folder");
        let path = dir.path().join("nested/frame-1.ppm");
        let rgba = [10u8, 20, 30, 255, 40, 50, 60, 255];
        write_ppm(&path, 2, 1, &rgba).expect("the frame is written");
        let bytes = std::fs::read(&path).expect("the frame is there");
        assert!(bytes.starts_with(b"P6\n2 1\n255\n"));
        assert_eq!(bytes.len(), b"P6\n2 1\n255\n".len() + 6);
        assert_eq!(&bytes[bytes.len() - 6..], [10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn formats_map_to_layouts() {
        assert_eq!(Pixels::of(28), Some(Pixels::Rgba8));
        assert_eq!(Pixels::of(87), Some(Pixels::Bgra8));
        assert_eq!(Pixels::of(24), Some(Pixels::Rgb10a2));
        assert_eq!(Pixels::of(10), None, "R16G16B16A16_FLOAT is not read");
        assert_eq!(Pixels::of(0), None);
    }
}
