use arrayvec::ArrayVec;
use bulletformat::{BulletFormat, ChessBoard};
use std::{
    sync::mpsc::SyncSender,
    thread::{self, JoinHandle},
};
pub const STM: usize = 0;
pub const XSTM: usize = 1;

pub const RECIPROCAL_SCALE: f32 = 1. / SCALE;
pub const SCALE: f32 = 400.;
pub const ALPHA: f32 = 0.10;

#[derive(Clone, Default, Debug)]
pub struct FeatureSet {
    pub features: [ArrayVec<usize, 32>; 2],
    pub blended_score: f32,
}

pub struct Dataloader {
    raw_data: Vec<ChessBoard>,
    _threads: usize,
}

impl Dataloader {
    pub fn new(threads: usize, raw_data: Vec<ChessBoard>) -> Self {
        Self {
            raw_data,
            _threads: threads,
        }
    }

    pub fn run(
        &self,
        sender: SyncSender<Vec<FeatureSet>>,
        mini_batch_size: usize,
        num_batches: usize,
    ) -> JoinHandle<()> {
        let c = self.raw_data.clone();
        thread::spawn(move || {
            let mut iter = c.iter();

            for _ in 0..num_batches {
                let vec = iter
                    .by_ref()
                    .take(mini_batch_size)
                    .map(|board| {
                        let mut features = FeatureSet::default();
                        for (piece, square) in board.into_iter() {
                            let (stm_idx, xstm_idx) = feature_extract(piece, square);
                            features.features[STM].push(stm_idx);
                            features.features[XSTM].push(xstm_idx);
                        }
                        features.blended_score = board.blended_result(ALPHA, RECIPROCAL_SCALE);
                        features
                    })
                    .collect::<Vec<_>>();
                sender.send(vec).unwrap();
            }
        })
    }
}

pub fn feature_extract(piece: u8, square: u8) -> (usize, usize) {
    let c = usize::from(piece & 8 > 0);
    let pc = 64 * usize::from(piece & 7);
    let sq = usize::from(square);
    let stm_idx = [0, 384][c] + pc + sq;
    let xstm_idx = [384, 0][c] + pc + (sq ^ 56);
    (stm_idx, xstm_idx)
}
