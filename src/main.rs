/// TODO: Figure out how tf to adjust LR properly
use bullet::format::{BulletFormat, ChessBoard};
use network::Network;
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    mem::transmute,
};

mod accumulator;
mod matrix;
mod network;
mod vector;

fn main() {
    const MEM_LIMIT: usize = 3000 * 1024 * 1024;
    let mut data = vec![];
    let mut buffer = [0u8; 32];
    let file_name = File::open("/home/jeff/chess-trainer/bruh.bin").unwrap();
    let mut fp = BufReader::new(&file_name);
    loop {
        fp.read_exact(&mut buffer).unwrap();
        data.push(unsafe { transmute::<[u8; 32], ChessBoard>(buffer) });
        if data.len() > 1_000_000 {
            break;
        }
        if data.len() * size_of::<ChessBoard>() > MEM_LIMIT {
            break;
        }
    }
    let mut net = Network::randomized();
    net.train(&mut data, 50, 1, 0.000_1);
    let mut out = File::open("./network.bin").unwrap();
    let buf: [u8; size_of::<Network>()] = unsafe { transmute(net) };
    let _ = out.write(&buf);
}
