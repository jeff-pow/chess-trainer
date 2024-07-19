use crate::{
    dataloader::{feature_extract, STM, XSTM},
    network::{activate, Network, HIDDEN_SIZE},
};
use bulletformat::ChessBoard;

#[derive(Copy, Clone)]
/// Struct to preserve the state of the first hidden layer of the network. Not really necessary
/// here like it is in a chess engine, but I find it makes it easier to think about and take
/// advantage of the sparse nature of the input layer
pub struct Accumulator {
    pub data: [[f32; HIDDEN_SIZE]; 2],
}

impl Accumulator {
    pub fn new(net: &Network) -> Self {
        Self {
            data: [net.feature_bias; 2],
        }
    }

    pub fn zeroed() -> Self {
        Self {
            data: [[0.; HIDDEN_SIZE]; 2],
        }
    }

    pub fn from_board(net: &Network, board: &ChessBoard) -> Self {
        let mut acc = Self::new(net);
        for (piece, square) in board.into_iter() {
            let (stm_idx, xstm_idx) = feature_extract(piece, square);
            acc.add(net, stm_idx, xstm_idx);
        }
        acc
    }

    pub fn add(&mut self, net: &Network, stm_feature_idx: usize, xstm_feature_idx: usize) {
        self.data[STM]
            .iter_mut()
            .zip(&net.feature_weights[stm_feature_idx])
            .for_each(|(x, &w)| *x += w);
        self.data[XSTM]
            .iter_mut()
            .zip(&net.feature_weights[xstm_feature_idx])
            .for_each(|(x, &w)| *x += w);
    }

    pub fn flatten(&self, net: &Network) -> f32 {
        let mut sum = net.output_bias;
        self.data[STM]
            .iter()
            .zip(&net.output_weights[STM])
            .for_each(|(&x, &w)| sum += activate(x) * w);
        self.data[XSTM]
            .iter()
            .zip(&net.output_weights[XSTM])
            .for_each(|(&x, &w)| sum += activate(x) * w);
        sum
    }
}
