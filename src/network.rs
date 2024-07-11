use crate::dataloader::Dataloader;
use bullet::{
    format::{BulletFormat, ChessBoard},
    inputs::{Chess768, InputType},
};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::{
    fs::File,
    io::Write,
    mem::transmute,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::sync_channel,
        Arc,
    },
    time::Instant,
};

pub const HIDDEN_SIZE: usize = 16;
pub const CLIP_VALUE: f32 = 100.0; // Gradient clipping threshold
pub const WEIGHT_CLIP: f32 = 1.98;

pub type FeatureVecs = ([[f32; 768]; 2], f32);

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(align(64))]
pub struct Network {
    pub feature_weights: [[f32; HIDDEN_SIZE]; 768],
    pub feature_bias: [f32; HIDDEN_SIZE],
    pub output_weights: [[f32; HIDDEN_SIZE]; 2],
    pub output_bias: f32,
}

impl Network {
    pub fn randomized() -> Self {
        let mut rng: StdRng = SeedableRng::seed_from_u64(0xABBA);
        // Sort of implements Xavier initialization, but I haven't given the actual arguments to
        // each too much thought. Sue me.
        let mut gen = |x: usize| rng.gen_range(-1. / (x as f32).sqrt()..1. / (x as f32).sqrt());

        let mut feature_weights = [[0f32; HIDDEN_SIZE]; 768];
        for a in &mut feature_weights {
            for b in a {
                *b = gen(768);
            }
        }
        let mut feature_bias = [0f32; HIDDEN_SIZE];
        for x in &mut feature_bias {
            *x = gen(HIDDEN_SIZE);
        }
        let mut output_weights = [[0f32; HIDDEN_SIZE]; 2];
        for x in &mut output_weights {
            for y in x {
                *y = gen(HIDDEN_SIZE);
            }
        }

        let output_bias = gen(1);
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
        mut lr: f32,
    ) {
        let mut rng: StdRng = SeedableRng::seed_from_u64(0xBEEF);
        for epoch in 0..epochs {
            if epoch % 10 == 0 && epoch != 0 {
                lr *= 0.5;
                let mut out = File::create("./network-checkpoint.bin").unwrap();
                let buf: [u8; size_of::<Network>()] = unsafe { transmute(*self) };
                let _ = out.write(&buf);
            }
            training_data.shuffle(&mut rng);
            let (sender, receiver) = sync_channel::<Vec<FeatureVecs>>(256);
            let loader = Dataloader::new(1, training_data.to_vec());
            let stop = Arc::new(AtomicBool::new(false));
            let handle = loader.run(sender, mini_batch_size, stop.clone());
            let num_chunks = training_data.len() / mini_batch_size;

            let start = Instant::now();
            for count in 0..num_chunks {
                let data = receiver.recv().unwrap();
                print!(
                    "\rEpoch {epoch} {:.1}% finished. {} pos / sec",
                    count as f32 * mini_batch_size as f32 / training_data.len() as f32 * 100.,
                    count as f64 * mini_batch_size as f64 / start.elapsed().as_secs_f64()
                );
                self.update_mini_batch(&data, lr);
            }
            println!();
            stop.store(true, Ordering::Relaxed);
            drop(handle);

            println!("Epoch: {}, Cost: {:.2}", epoch, self.cost(training_data));
        }
    }

    fn update_mini_batch(&mut self, batch: &[FeatureVecs], lr: f32) {
        let mut unified_changes = Network::zeroed();
        batch.iter().for_each(|board| {
            self.backward(board, &mut unified_changes);
        });

        self.apply_gradients(&unified_changes, lr, batch.len() as f32);
    }

    fn apply_gradients(&mut self, changes: &Network, lr: f32, batch_size: f32) {
        let lr_batch = lr / batch_size;

        self.feature_weights
            .iter_mut()
            .flatten()
            .zip(changes.feature_weights.iter().flatten())
            .for_each(|(w, nw)| {
                *w -= lr_batch * nw;
                *w = w.clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
            });

        self.feature_bias
            .iter_mut()
            .zip(changes.feature_bias.iter())
            .for_each(|(b, nb)| {
                *b -= lr_batch * nb;
                *b = b.clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
            });

        self.output_weights
            .iter_mut()
            .flatten()
            .zip(changes.output_weights.iter().flatten())
            .for_each(|(w, nw)| {
                *w -= lr_batch * nw;
                *w = w.clamp(-WEIGHT_CLIP, WEIGHT_CLIP);
            });

        self.output_bias -= lr_batch * changes.output_bias;
        self.output_bias = self.output_bias.clamp(-WEIGHT_CLIP, WEIGHT_CLIP);

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

    fn backward(&self, board: &FeatureVecs, deltas: &mut Self) {
        let input_layer = board.0;

        let mut hl = [self.feature_bias; 2];
        sparse_matmul(&self.feature_weights, &input_layer, &mut hl);
        // for i in 0..768 {
        //     for j in 0..HIDDEN_SIZE {
        //         hl[STM][j] += self.feature_weights[i][j] * input_layer[STM][i];
        //         hl[XSTM][j] += self.feature_weights[i][j] * input_layer[XSTM][i];
        //     }
        // }
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

        //
        // Backwards Pass
        //
        let output_delta = (output_activate - board.1) * output_z;
        deltas.output_bias += output_delta;

        for i in 0..HIDDEN_SIZE {
            deltas.output_weights[STM][i] += output_delta * hl_activate[STM][i];
            deltas.output_weights[XSTM][i] += output_delta * hl_activate[XSTM][i];
        }

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
                deltas.feature_weights[i][j] += delta_dot_sp[STM][j] * input_layer[STM][i];
                deltas.feature_weights[i][j] += delta_dot_sp[XSTM][j] * input_layer[XSTM][i];
            }
        }

        for i in 0..HIDDEN_SIZE {
            deltas.feature_bias[i] += delta_dot_sp[STM][i] + delta_dot_sp[XSTM][i];
        }
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
