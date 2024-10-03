use arrayvec::ArrayVec;
use bulletformat::{BulletFormat, ChessBoard};
use std::{
    fs::File,
    io::{self, Read, Seek},
    slice::from_raw_parts,
    sync::mpsc::SyncSender,
    thread::{self, JoinHandle},
};
pub const STM: usize = 0;
pub const XSTM: usize = 1;

pub const RECIPROCAL_SCALE: f32 = 1. / SCALE;
pub const SCALE: f32 = 400.;
pub const ALPHA: f32 = 0.90;

#[derive(Clone, Default, Debug)]
pub struct FeatureSet {
    pub features: [ArrayVec<usize, 32>; 2],
    pub blended_score: f32,
}

pub struct Dataloader {}

impl Dataloader {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(
        &self,
        sender: SyncSender<Vec<FeatureSet>>,
        mini_batch_size: usize,
        num_batches: usize,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            let mut file_name = File::open("/home/jeff/chess-trainer/shuffled.bin").unwrap();
            let mut buffer = vec![0u8; mini_batch_size * size_of::<ChessBoard>()];
            let mut count = 0;
            while count < num_batches {
                match file_name.read_exact(&mut buffer) {
                    Ok(_) => {
                        let slice = unsafe {
                            from_raw_parts(buffer.as_ptr() as *const ChessBoard, mini_batch_size)
                        };
                        assert_eq!(mini_batch_size, slice.len());
                        let vec = slice
                            .iter()
                            .map(|board| {
                                let mut features = FeatureSet::default();
                                for (piece, square) in board.into_iter() {
                                    let (stm_idx, xstm_idx) = feature_extract(piece, square);
                                    features.features[STM].push(stm_idx);
                                    features.features[XSTM].push(xstm_idx);
                                }
                                features.blended_score =
                                    board.blended_result(ALPHA, RECIPROCAL_SCALE);
                                features
                            })
                            .collect::<Vec<_>>();
                        count += 1;
                        sender.send(vec).unwrap();
                    }
                    Err(e) => {
                        if e.kind() == io::ErrorKind::UnexpectedEof {
                            println!("Seeked");
                            file_name.seek(io::SeekFrom::Start(0)).unwrap();
                        } else {
                            panic!("Could not rewind data file.");
                        }
                    }
                }
            }
        })
    }
}

pub fn feature_extract(piece: u8, square: u8) -> (usize, usize) {
    let c = usize::from(piece & 8 > 0);
    let pc = 64 * usize::from(piece & 7);
    let sq = usize::from(square);
    let stm_idx = [0, 384][c] + pc + sq;
    let xstm_idx = [384, 0][c] + pc + (sq ^ 56);
    (stm_idx, xstm_idx)
}
