use crate::{
    accumulator::Accumulator,
    dataloader::{Dataloader, FeatureSet, SCALE, STM, XSTM},
};
use bulletformat::ChessBoard;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::{
    fs::File, io::Write, mem::transmute, str::FromStr, sync::mpsc::sync_channel, time::Instant,
};

pub const HIDDEN_SIZE: usize = 32;
pub const WEIGHT_CLIP: f32 = 1.98;

#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(C, align(64))]
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
                *b = gen(768 * HIDDEN_SIZE);
            }
        }
        let mut feature_bias = [0f32; HIDDEN_SIZE];
        for x in &mut feature_bias {
            *x = gen(HIDDEN_SIZE);
        }
        let mut output_weights = [[0f32; HIDDEN_SIZE]; 2];
        for x in &mut output_weights {
            for y in x {
                *y = gen(HIDDEN_SIZE * 2);
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
                lr *= 0.7;
                let mut out = File::create("./network-checkpoint.bin").unwrap();
                let buf: [u8; size_of::<Network>()] = unsafe { transmute(*self) };
                let _ = out.write(&buf);
            }
            training_data.shuffle(&mut rng);

            let (sender, receiver) = sync_channel::<Vec<FeatureSet>>(256);
            let loader = Dataloader::new(1, training_data.to_vec());
            let num_chunks = training_data.len() / mini_batch_size;
            let mut error = 0.;
            let data_loader_handle = loader.run(sender.clone(), mini_batch_size, num_chunks);

            let start = Instant::now();
            for count in 0..num_chunks {
                let data = receiver.recv().unwrap();
                print!(
                    "\rEpoch {epoch} {:.1}% finished. {:.0} pos / sec",
                    count as f32 * mini_batch_size as f32 / training_data.len() as f32 * 100.,
                    count as f32 * mini_batch_size as f32 / start.elapsed().as_secs_f32()
                );
                self.update_mini_batch(&data, lr, &mut error);
            }
            println!();
            data_loader_handle.join().unwrap();
            let start_pos = ChessBoard::from_str(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 33 | 0.5",
            )
            .unwrap();

            println!(
                "Epoch: {}, Error: {:.6}, Evaluation: {}",
                epoch,
                error / training_data.len() as f64,
                self.evaluate(&start_pos)
            );
        }
    }

    fn update_mini_batch(&mut self, batch: &[FeatureSet], lr: f32, error: &mut f64) {
        let mut unified_changes = Network::zeroed();
        batch.iter().for_each(|board| {
            self.backward(board, &mut unified_changes, error);
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
    }

    pub fn print_max_and_min(&self) {
        let mut feature_weight_max = f32::MIN;
        let mut feature_weight_min = f32::MAX;
        for &w in self.feature_weights.iter().flatten() {
            feature_weight_min = w.min(feature_weight_min);
            feature_weight_max = w.max(feature_weight_max);
        }

        let mut feature_bias_max = f32::MIN;
        let mut feature_bias_min = f32::MAX;
        for &w in self.feature_bias.iter() {
            feature_bias_min = w.min(feature_bias_min);
            feature_bias_max = w.max(feature_bias_max);
        }

        let mut output_weight_max = f32::MIN;
        let mut output_weight_min = f32::MAX;
        for &w in self.output_weights.iter().flatten() {
            output_weight_min = w.min(output_weight_min);
            output_weight_max = w.max(output_weight_max);
        }

        dbg!(
            feature_weight_min,
            feature_weight_max,
            feature_bias_min,
            feature_bias_max,
            output_weight_min,
            output_weight_max,
            self.output_bias
        );
    }

    fn backward(&self, board: &FeatureSet, deltas: &mut Self, error: &mut f64) {
        let input_layer = &board.features;

        let mut hl = Accumulator::new(self);
        for (&stm_idx, &xstm_idx) in board.features[STM].iter().zip(&board.features[XSTM]) {
            hl.add(self, stm_idx, xstm_idx);
        }

        let mut hl_activated = [[0f32; HIDDEN_SIZE]; 2];
        for (a, &x) in hl_activated
            .iter_mut()
            .flatten()
            .zip(hl.data.iter().flatten())
        {
            *a = activate(x);
        }

        let mut eval = self.output_bias;
        for i in 0..HIDDEN_SIZE {
            eval += self.output_weights[STM][i] * hl_activated[STM][i];
            eval += self.output_weights[XSTM][i] * hl_activated[XSTM][i];
        }

        //
        // Backwards Pass
        //
        let sigmoid = 1.0 / (1.0 + (-eval).exp());
        let diff = sigmoid - board.blended_score;
        let output_delta = diff * sigmoid * (1.0 - sigmoid);

        *error += f64::from(diff.powi(2));
        deltas.output_bias += output_delta;

        for i in 0..HIDDEN_SIZE {
            deltas.output_weights[STM][i] += output_delta * hl_activated[STM][i];
            deltas.output_weights[XSTM][i] += output_delta * hl_activated[XSTM][i];
        }

        let mut sp = Accumulator::zeroed();
        for (s, &h) in sp.data.iter_mut().flatten().zip(hl.data.iter().flatten()) {
            *s = prime(h);
        }

        let delta_dot_sp = {
            let mut arr = sp;
            for i in 0..HIDDEN_SIZE {
                arr.data[STM][i] *= output_delta * self.output_weights[STM][i];
                arr.data[XSTM][i] *= output_delta * self.output_weights[XSTM][i];
            }
            arr
        };

        for &i in &input_layer[STM] {
            for j in 0..HIDDEN_SIZE {
                deltas.feature_weights[i][j] += delta_dot_sp.data[STM][j];
            }
        }
        for &i in &input_layer[XSTM] {
            for j in 0..HIDDEN_SIZE {
                deltas.feature_weights[i][j] += delta_dot_sp.data[XSTM][j];
            }
        }

        for i in 0..HIDDEN_SIZE {
            deltas.feature_bias[i] += delta_dot_sp.data[STM][i] + delta_dot_sp.data[XSTM][i];
        }
    }

    /// Feed forward evaluation including wdl scale
    pub fn evaluate(&self, board: &ChessBoard) -> f32 {
        let acc = Accumulator::from_board(self, board);
        acc.flatten(self) * SCALE
    }
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
