use arrayvec::ArrayVec;
use bullet::{
    format::{BulletFormat, ChessBoard},
    inputs::{Chess768, InputType},
};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
        Arc,
    },
    thread::{self, JoinHandle},
};
pub const STM: usize = 0;
pub const XSTM: usize = 1;

pub const SCALE: f32 = 400.;
pub const ALPHA: f32 = 0.10;

#[derive(Clone, Default)]
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
        stop: Arc<AtomicBool>,
        num_batches: usize,
    ) -> JoinHandle<()> {
        let c = self.raw_data.clone();
        thread::spawn(move || {
            let mut iter = c.iter();

            for _ in 0..num_batches {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                let vec = iter
                    .by_ref()
                    .take(mini_batch_size)
                    .map(|board| {
                        let mut features = FeatureSet::default();
                        let chess_768 = Chess768;
                        for (stm_idx, xstm_idx) in chess_768.feature_iter(board) {
                            features.features[STM].push(stm_idx);
                            features.features[XSTM].push(xstm_idx);
                        }
                        features.blended_score = board.blended_result(ALPHA, SCALE);
                        features
                    })
                    .collect::<Vec<_>>();
                sender.send(vec).unwrap();
            }
        })
    }
}
