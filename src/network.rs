use bullet::{
    format::{BulletFormat, ChessBoard},
    inputs::{Chess768, InputType},
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

pub const HIDDEN_SIZE: usize = 16;
pub const L2_REGULARIZATION: f32 = 0.01; // L2 regularization factor
pub const CLIP_VALUE: f32 = 1.0; // Gradient clipping threshold

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Network {
    pub feature_weights: [[f32; HIDDEN_SIZE]; 768],
    pub feature_bias: [f32; HIDDEN_SIZE],
    pub output_weights: [[f32; HIDDEN_SIZE]; 2],
    pub output_bias: f32,
}

impl Network {
    pub fn randomized() -> Self {
        let mut rng: StdRng = SeedableRng::seed_from_u64(0xABBA);

        let mut feature_weights = [[0f32; HIDDEN_SIZE]; 768];
        for a in &mut feature_weights {
            for b in a {
                *b = rng.gen();
            }
        }
        let mut feature_bias = [0f32; HIDDEN_SIZE];
        for x in &mut feature_bias {
            *x = rng.gen();
        }
        let mut output_weights = [[0f32; HIDDEN_SIZE]; 2];
        for x in &mut output_weights {
            for y in x {
                *y = rng.gen();
            }
        }

        let output_bias = rng.gen();
        Self {
            feature_weights,
            feature_bias,
            output_weights,
            output_bias,
        }
    }

    pub fn zeroed() -> Self {
        Self {
            feature_weights: [[0f32; HIDDEN_SIZE]; 768],
            feature_bias: [0f32; HIDDEN_SIZE],
            output_weights: [[0f32; HIDDEN_SIZE]; 2],
            output_bias: 0f32,
        }
    }

    pub fn train(
        &mut self,
        training_data: &mut [ChessBoard],
        epochs: usize,
        mini_batch_size: usize,
        lr: f32,
    ) {
        let mut rng: StdRng = SeedableRng::seed_from_u64(0xBEEF);
        for i in 0..epochs {
            training_data.shuffle(&mut rng);
            for (count, data) in training_data.chunks(mini_batch_size).enumerate() {
                assert!(self.is_ok());
                self.update_mini_batch(data, lr);
            }

            assert_ne!(*self, Self::randomized());
            println!("Epoch: {}, Loss: {:.2}%", i, self.loss(training_data));
        }
    }

    pub fn is_ok(&self) -> bool {
        for x in self.feature_weights.iter().flatten() {
            if !x.is_finite() {
                return false;
            }
        }
        for x in self.feature_bias.iter() {
            if !x.is_finite() {
                return false;
            }
        }
        for x in self.output_weights.iter().flatten() {
            if !x.is_finite() {
                return false;
            }
        }
        self.output_bias.is_finite()
    }

    fn update_mini_batch(&mut self, batch: &[ChessBoard], lr: f32) {
        let mut changes = Network::zeroed();

        for (idx, board) in batch.iter().enumerate() {
            let deltas = self.backward(board);
            assert!(deltas.is_ok());

            changes
                .feature_weights
                .iter_mut()
                .flatten()
                .zip(deltas.feature_weights.iter().flatten())
                .for_each(|(w, nw)| *w += nw);
            changes
                .feature_bias
                .iter_mut()
                .zip(deltas.feature_bias.iter())
                .for_each(|(w, nw)| *w += nw);
            changes
                .output_weights
                .iter_mut()
                .flatten()
                .zip(deltas.output_weights.iter().flatten())
                .for_each(|(w, nw)| *w += nw);
            changes.output_bias += deltas.output_bias;
        }

        self.apply_gradients(&changes, lr, batch.len() as f32);
        assert!(changes.is_ok());
    }

    fn apply_gradients(&mut self, changes: &Network, lr: f32, batch_size: f32) {
        let lr_batch = lr / batch_size;

        self.feature_weights
            .iter_mut()
            .flatten()
            .zip(changes.feature_weights.iter().flatten())
            .for_each(|(w, nw)| {
                *w -= lr_batch * nw;
                *w -= lr * L2_REGULARIZATION * *w;
            });

        self.feature_bias
            .iter_mut()
            .zip(changes.feature_bias.iter())
            .for_each(|(b, nb)| {
                *b -= lr_batch * nb;
                *b -= lr * L2_REGULARIZATION * *b;
            });

        self.output_weights
            .iter_mut()
            .flatten()
            .zip(changes.output_weights.iter().flatten())
            .for_each(|(w, nw)| {
                *w -= lr_batch * nw;
                *w -= lr * L2_REGULARIZATION * *w;
            });

        self.output_bias -= lr_batch * changes.output_bias;
        self.output_bias -= lr * L2_REGULARIZATION * self.output_bias;

        // Gradient clipping
        self.clip_gradients();
    }

    fn clip_gradients(&mut self) {
        self.feature_weights
            .iter_mut()
            .flatten()
            .for_each(|w| *w = w.clamp(-CLIP_VALUE, CLIP_VALUE));

        self.feature_bias
            .iter_mut()
            .for_each(|b| *b = b.clamp(-CLIP_VALUE, CLIP_VALUE));

        self.output_weights
            .iter_mut()
            .flatten()
            .for_each(|w| *w = w.clamp(-CLIP_VALUE, CLIP_VALUE));

        self.output_bias = self.output_bias.clamp(-CLIP_VALUE, CLIP_VALUE);
    }

    fn backward(&self, board: &ChessBoard) -> Self {
        let input_layer = extract_features(board);

        let mut hl = [self.feature_bias; 2];
        for i in 0..768 {
            for j in 0..HIDDEN_SIZE {
                hl[STM][j] += self.feature_weights[i][j] * input_layer[STM][i];
                hl[XSTM][j] += self.feature_weights[i][j] * input_layer[XSTM][i];
            }
        }
        let hl_z = hl;
        let hl_activate = {
            let mut arr = hl;
            arr.iter_mut().flatten().for_each(|x| *x = activate(*x));
            arr
        };

        let mut output_z = self.output_bias;
        for i in 0..HIDDEN_SIZE {
            output_z += self.output_weights[STM][i] * hl_activate[STM][i];
            output_z += self.output_weights[XSTM][i] * hl_activate[XSTM][i];
        }
        let output_activate = output_z;
        assert!(self.is_ok());

        let mut deltas = Network::zeroed();

        //
        // Backwards Pass
        //
        let output_delta = (output_activate - board.score() as f32) * output_z;
        deltas.output_bias = output_delta;
        assert!(deltas.is_ok());

        for i in 0..HIDDEN_SIZE {
            deltas.output_weights[STM][i] = output_delta * hl_activate[STM][i];
            deltas.output_weights[XSTM][i] = output_delta * hl_activate[XSTM][i];
        }
        assert!(deltas.is_ok());

        let z = hl_z;
        let sp = {
            let mut arr = z;
            arr.iter_mut().flatten().for_each(|x| *x = prime(*x));
            arr
        };
        let delta_dot_sp = {
            let mut arr = sp;
            for i in 0..HIDDEN_SIZE {
                arr[STM][i] *= output_delta * self.output_weights[STM][i];
                arr[XSTM][i] *= output_delta * self.output_weights[XSTM][i];
            }
            arr
        };

        for i in 0..768 {
            for j in 0..HIDDEN_SIZE {
                deltas.feature_weights[i][j] = delta_dot_sp[STM][j] * input_layer[STM][i];
                deltas.feature_weights[i][j] += delta_dot_sp[XSTM][j] * input_layer[XSTM][i];
            }
        }
        assert!(deltas.is_ok());

        for i in 0..HIDDEN_SIZE {
            deltas.feature_bias[i] = delta_dot_sp[STM][i] + delta_dot_sp[XSTM][i];
        }

        assert_ne!(deltas, Self::zeroed());
        assert!(deltas.is_ok());

        deltas
    }

    // Double vertical bars denoting the magnitude of the vector
    /// Cost = 1 / 2n * sum (||(prediction - actual)|| ** 2)
    pub fn cost(&self, test_data: &[ChessBoard]) -> f32 {
        0.5 * test_data
            .iter()
            .map(|board| {
                let y = board.score() as f32;
                let prediction = self.feed_forward(board);
                (y - prediction).powi(2)
            })
            .sum::<f32>()
            / test_data.len() as f32
    }

    pub fn loss(&self, test_data: &[ChessBoard]) -> f32 {
        0.5 * test_data
            .iter()
            .map(|board| {
                let y = board.score() as f32;
                let prediction = self.feed_forward(board);
                (y - prediction).powi(2)
            })
            .sum::<f32>()
    }

    pub fn feed_forward(&self, data: &ChessBoard) -> f32 {
        let features = extract_features(data);
        //hl = b
        let mut hl = [self.feature_bias; 2];
        // hl = wa + b
        sparse_matmul(&self.feature_weights, &features, &mut hl);
        // hl = activated(wa + b)
        hl.iter_mut().flatten().for_each(|x| *x = activate(*x));
        // sum = b
        let mut sum = self.output_bias;
        // sum = wa + b
        for i in 0..HIDDEN_SIZE {
            sum += hl[0][i] * self.output_weights[0][i];
            sum += hl[1][i] * self.output_weights[1][i];
        }
        sum
    }
}

fn sparse_matmul(
    weights: &[[f32; HIDDEN_SIZE]; 768],
    feature_vec: &[[f32; 768]; 2],
    dest: &mut [[f32; HIDDEN_SIZE]; 2],
) {
    for i in 0..768 {
        if feature_vec[0][i] != 0. {
            for j in 0..HIDDEN_SIZE {
                dest[0][j] += feature_vec[0][i] * weights[i][j]
            }
        }
    }
    for i in 0..768 {
        if feature_vec[1][i] != 0. {
            for j in 0..HIDDEN_SIZE {
                dest[1][j] += feature_vec[1][i] * weights[i][j]
            }
        }
    }
}

const STM: usize = 0;
const XSTM: usize = 1;

pub fn extract_features(data: &ChessBoard) -> [[f32; 768]; 2] {
    let mut features = [[0f32; 768]; 2];
    let chess_768 = Chess768;
    for (stm_idx, xstm_idx) in chess_768.feature_iter(data) {
        features[STM][stm_idx] = 1.;
        features[XSTM][xstm_idx] = 1.;
    }
    features
}

pub fn activate(x: f32) -> f32 {
    x.clamp(0., 1.).powi(2)
}

pub fn prime(x: f32) -> f32 {
    if x > 0.0 && x < 1.0 {
        2.0 * x
    } else {
        0.0
    }
}
