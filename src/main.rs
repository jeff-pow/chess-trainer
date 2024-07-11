use bullet::format::ChessBoard;
use network::Network;
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    mem::transmute,
    str::FromStr,
};

mod accumulator;
mod dataloader;
mod matrix;
mod network;
mod quantize;
mod utils;
mod vector;
pub const MEM_LIMIT: usize = 3000 * 1024 * 1024;

fn main() {
    // let net: Network = unsafe { transmute(*include_bytes!("../network.bin")) };
    let fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0";
    let b = ChessBoard::from_str(fen).unwrap();
    // dbg!(net.feed_forward(&b));

    let mut data = vec![];
    let mut buffer = [0u8; 32];
    let file_name = File::open("/home/jeff/chess-trainer/bruh.bin").unwrap();
    let mut fp = BufReader::new(&file_name);
    loop {
        fp.read_exact(&mut buffer).unwrap();
        data.push(unsafe { transmute::<[u8; 32], ChessBoard>(buffer) });
        if data.len() > 16_000_000 {
            break;
        }
        if data.len() * size_of::<ChessBoard>() > MEM_LIMIT {
            break;
        }
    }

    let mut net = Network::randomized();
    net.train(&mut data, 50, 100, 0.1);
    let mut out = File::create("./network.bin").unwrap();
    let buf: [u8; size_of::<Network>()] = unsafe { transmute(net) };
    let _ = out.write(&buf);
}
