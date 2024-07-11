use crate::network::{extract_features, FeatureVecs};
use bullet::format::{BulletFormat, ChessBoard};
use std::{
    sync::mpsc::SyncSender,
    thread::{self, JoinHandle},
};

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
        sender: SyncSender<Vec<FeatureVecs>>,
        mini_batch_size: usize,
    ) -> JoinHandle<()> {
        let c = self.raw_data.clone();
        thread::spawn(move || {
            let mut iter = c.iter().cycle();
            loop {
                let vec = iter
                    .by_ref()
                    .take(mini_batch_size)
                    .map(|b| (extract_features(b), b.score() as f32))
                    .collect::<Vec<_>>();
                while sender.send(vec.clone()).is_err() {}
            }
        })
    }
}
