use bulletformat::ChessBoard;
use network::Network;
use quantize::QuantizedNetwork;
use std::{fs::File, io::Write, mem::transmute, str::FromStr};

mod accumulator;
mod dataloader;
mod matrix;
mod network;
mod quantize;
mod utils;
mod vector;

fn main() {
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 33 | 0.5";
    let start_pos = ChessBoard::from_str(fen).unwrap();

    let mut net = Network::randomized();
    net.train(32_000_000, 100, 100, 0.001);
    net.print_max_and_min();

    let mut out = File::create("./network.bin").unwrap();
    let buf: [u8; size_of::<Network>()] = unsafe { transmute(net) };
    let _ = out.write(&buf);

    dbg!(net.evaluate(&start_pos));
    let quantized = QuantizedNetwork::new(&net);
    quantized.write("./quantized-network.bin");
}
