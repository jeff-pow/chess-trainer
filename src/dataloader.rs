use crate::network::extract_features;
use bullet::format::ChessBoard;
use std::{
    sync::mpsc::Sender,
    thread::{self, JoinHandle},
};

pub struct Dataloader {
    raw_data: Vec<ChessBoard>,
    threads: usize,
}

impl Dataloader {
    pub fn new(threads: usize, raw_data: Vec<ChessBoard>) -> Self {
        Self { raw_data, threads }
    }

    pub fn run(&self, sender: Sender<[[f32; 768]; 2]>) -> JoinHandle<_> {
        thread::spawn(move || {
            let iter = self.raw_data.iter();
            loop {
                let features = extract_features(iter.next());
                while sender.send(features).is_err() {}
            }
        })
    }
}
