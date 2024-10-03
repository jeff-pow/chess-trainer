use crate::network::{Network, HIDDEN_SIZE};
use std::{fs::File, io::Write, mem::transmute};

const QA: i32 = 255;
const QB: i32 = 64;
const QAB: i32 = QA * QB;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct QuantizedNetwork {
    pub feature_weights: [[i16; HIDDEN_SIZE]; 768],
    pub feature_bias: [i16; HIDDEN_SIZE],
    pub output_weights: [[i16; HIDDEN_SIZE]; 2],
    pub output_bias: i16,
}

impl QuantizedNetwork {
    pub fn new(net: &Network) -> Self {
        let mut ret = Self {
            feature_weights: [[0; HIDDEN_SIZE]; 768],
            feature_bias: [0; HIDDEN_SIZE],
            output_weights: [[0; HIDDEN_SIZE]; 2],
            output_bias: 0,
        };

        for (&p, r) in net
            .feature_weights
            .iter()
            .flatten()
            .zip(ret.feature_weights.iter_mut().flatten())
        {
            *r = (p * QA as f32) as i16;
        }

        for (&p, r) in net.feature_bias.iter().zip(ret.feature_bias.iter_mut()) {
            *r = (p * QA as f32) as i16;
        }

        for (&p, r) in net
            .output_weights
            .iter()
            .flatten()
            .zip(ret.output_weights.iter_mut().flatten())
        {
            *r = (p * QB as f32) as i16;
        }

        ret.output_bias = (net.output_bias * QAB as f32) as i16;

        ret
    }

    pub fn write(&self, file_name: &str) {
        let mut out = File::create(file_name).unwrap();
        let buf: [u8; size_of::<QuantizedNetwork>()] = unsafe { transmute(*self) };
        let _ = out.write(&buf);
    }
}
