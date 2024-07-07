use crate::network::{activate, Network, HIDDEN_SIZE};
use bullet::{
    format::ChessBoard,
    inputs::{Chess768, InputType},
};

#[derive(Copy, Clone)]
/// Struct to preserve the state of the first hidden layer of the network. Not really necessary
/// here like it is in a chess engine, but I find it makes it easier to think about and take
/// advantage of the sparse nature of the input layer
pub struct Accumulator {
    pub stm: [f32; HIDDEN_SIZE],
    pub xstm: [f32; HIDDEN_SIZE],
}

impl Accumulator {
    pub fn new(net: &Network) -> Self {
        Self {
            stm: net.feature_bias,
            xstm: net.feature_bias,
        }
    }

    pub fn from_board(net: &Network, board: &ChessBoard) -> Self {
        let mut acc = Self::new(net);
        let chess_768 = Chess768;
        for (stm_idx, xstm_idx) in chess_768.feature_iter(board) {
            acc.add(net, stm_idx, xstm_idx);
        }
        acc
    }

    pub fn add(&mut self, net: &Network, stm_feature_idx: usize, xstm_feature_idx: usize) {
        self.stm
            .iter_mut()
            .zip(&net.feature_weights[stm_feature_idx])
            .for_each(|(x, &w)| *x += w);
        self.xstm
            .iter_mut()
            .zip(&net.feature_weights[xstm_feature_idx])
            .for_each(|(x, &w)| *x += w);
    }

    pub fn flatten(&self, net: &Network) -> f32 {
        let mut sum = net.output_bias;
        self.stm
            .iter()
            .zip(&net.output_weights[0])
            .for_each(|(&x, &w)| sum += activate(x) * w);
        self.xstm
            .iter()
            .zip(&net.output_weights[1])
            .for_each(|(&x, &w)| sum += activate(x) * w);
        sum
    }
}
