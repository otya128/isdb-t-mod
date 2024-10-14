use reed_solomon::Encoder;
use rustfft::{num_complex::Complex32, FftPlanner};
use std::{
    collections::VecDeque,
    io::{self, Read, Write},
    time::Instant,
};

const SEGMENTS: usize = 13;
const TS_SIZE: usize = 188;
const TS_PARITY_SIZE: usize = 16;
const TSP_SIZE: usize = TS_SIZE + TS_PARITY_SIZE;
const TS_SYNC_BYTE: u8 = 0x47;

const BYTE_PRBS_INITIAL_STATE: u16 = 0b100101010000000;
fn byte_prbs(mut d: u16) -> u16 {
    let d15 = d & 1;
    let d14 = (d & 2) >> 1;
    d >>= 1;
    if (d15 ^ d14) != 0 {
        d |= 0b100000000000000;
    }
    return d;
}
fn next_byte_prbs(byte_prbs_state: &mut u16) -> u8 {
    let mut b: u8 = 0;
    for bi in [1, 2, 4, 8, 16, 32, 64, 128].iter().rev() {
        *byte_prbs_state = byte_prbs(*byte_prbs_state);
        if *byte_prbs_state & 0b100000000000000 != 0 {
            b |= bi;
        }
    }
    return b;
}

struct ByteInterleaver {
    buffers: [VecDeque<u8>; 12],
    index: usize,
}

impl ByteInterleaver {
    fn new() -> Self {
        let buffers = [
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 0)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 1)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 2)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 3)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 4)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 5)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 6)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 7)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 8)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 9)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 10)),
            VecDeque::from_iter(std::iter::repeat(0).take(17 * 11)),
        ];
        return Self { buffers, index: 0 };
    }
    fn push(&mut self, byte: u8) -> u8 {
        let buffer = &mut self.buffers[self.index];
        buffer.push_back(byte);
        self.index = (self.index + 1) % 12;
        return buffer.pop_front().unwrap();
    }
}

struct TimeInterleaver {
    buffers: Vec<VecDeque<Complex32>>,
}

impl TimeInterleaver {
    fn new(length: usize, carriers: usize, delay: usize) -> Self {
        let mut buffers = Vec::with_capacity(carriers);
        for i in 0..carriers {
            let m_i = (i * 5) % 96;
            buffers.push(VecDeque::from_iter(
                std::iter::repeat(Complex32::ZERO).take(length * m_i + delay),
            ));
        }
        return Self { buffers };
    }
    fn push(&mut self, symbol: Complex32, carrier_index: usize) -> Complex32 {
        let buffer = &mut self.buffers[carrier_index];
        buffer.push_back(symbol);
        return buffer.pop_front().unwrap();
    }
}

fn convolve(
    data: &[u8],
    mut d: u8,
    puncture_pattern: &[bool],
    mut puncture_state: usize,
) -> (Vec<bool>, u8, usize) {
    let mut v = Vec::with_capacity(data.len() * 2);
    const G1: u8 = 0o171;
    const G2: u8 = 0o133;
    for b in data {
        for i in (0..8).rev() {
            if (b & (1 << i)) != 0 {
                d |= 0b1000000;
            }
            let c0 = ((d & G1).count_ones() & 1) == 1;
            let c1 = ((d & G2).count_ones() & 1) == 1;
            if puncture_pattern[puncture_state] {
                v.push(c0);
            }
            if puncture_pattern[puncture_state + 1] {
                v.push(c1);
            }
            puncture_state = (puncture_state + 2) % puncture_pattern.len();
            d >>= 1;
        }
    }
    return (v, d, puncture_state);
}

const PUNCTURE_1_2: [bool; 2] = [
    true, true, //
];

const PUNCTURE_2_3: [bool; 4] = [
    true, true, //
    false, true, //
];

const PUNCTURE_3_4: [bool; 6] = [
    true, true, //
    false, true, //
    true, false, //
];

const PUNCTURE_5_6: [bool; 10] = [
    true, true, //
    false, true, //
    true, false, //
    false, true, //
    true, false, //
];

const PUNCTURE_7_8: [bool; 14] = [
    true, true, //
    false, true, //
    false, true, //
    false, true, //
    true, false, //
    false, true, //
    true, false, //
];

const AC_CARRIER: &[&[usize]] = &[
    &[10, 28, 161, 191, 277, 316, 335, 425],
    &[20, 40, 182, 208, 251, 295, 400, 421],
    &[4, 89, 148, 197, 224, 280, 331, 413],
    &[98, 101, 118, 136, 269, 299, 385, 424],
    &[11, 101, 128, 148, 290, 316, 359, 403],
    &[76, 97, 112, 197, 256, 305, 332, 388],
    &[7, 89, 206, 209, 226, 244, 377, 407],
    &[61, 100, 119, 209, 236, 256, 398, 424],
    &[35, 79, 184, 205, 220, 305, 364, 413],
    &[8, 64, 115, 197, 314, 317, 334, 352],
    &[53, 83, 169, 208, 227, 317, 344, 364],
    &[74, 100, 143, 187, 292, 313, 328, 413],
    &[40, 89, 116, 172, 223, 305, 422, 425],
];

const TMCC_CARRIER: &[&[usize]] = &[
    &[70, 133, 233, 410],
    &[44, 155, 265, 355],
    &[83, 169, 301, 425],
    &[23, 178, 241, 341],
    &[86, 152, 263, 373],
    &[31, 191, 277, 409],
    &[101, 131, 286, 349],
    &[17, 194, 260, 371],
    &[49, 139, 299, 385],
    &[85, 209, 239, 394],
    &[25, 125, 302, 368],
    &[47, 157, 247, 407],
    &[61, 193, 317, 347],
];

#[allow(unused)]
const MODE1_CARRIER_RANDOMIZE: &[usize] = &[
    80, 93, 63, 92, 94, 55, 17, 81, 6, 51, 9, 85, 89, 65, 52, 15, 73, 66, 46, 71, 12, 70, 18, 13,
    95, 34, 1, 38, 78, 59, 91, 64, 0, 28, 11, 4, 45, 35, 16, 7, 48, 22, 23, 77, 56, 19, 8, 36, 39,
    61, 21, 3, 26, 69, 67, 20, 74, 86, 72, 25, 31, 5, 49, 42, 54, 87, 43, 60, 29, 2, 76, 84, 83,
    40, 14, 79, 27, 57, 44, 37, 30, 68, 47, 88, 75, 41, 90, 10, 33, 32, 62, 50, 58, 82, 53, 24,
];

#[allow(unused)]
const MODE2_CARRIER_RANDOMIZE: &[usize] = &[
    98, 35, 67, 116, 135, 17, 5, 93, 73, 168, 54, 143, 43, 74, 165, 48, 37, 69, 154, 150, 107, 76,
    176, 79, 175, 36, 28, 78, 47, 128, 94, 163, 184, 72, 142, 2, 86, 14, 130, 151, 114, 68, 46,
    183, 122, 112, 180, 42, 105, 97, 33, 134, 177, 84, 170, 45, 187, 38, 167, 10, 189, 51, 117,
    156, 161, 25, 89, 125, 139, 24, 19, 57, 71, 39, 77, 191, 88, 85, 0, 162, 181, 113, 140, 61, 75,
    82, 101, 174, 118, 20, 136, 3, 121, 190, 120, 92, 160, 52, 153, 127, 65, 60, 133, 147, 131, 87,
    22, 58, 100, 111, 141, 83, 49, 132, 12, 155, 146, 102, 164, 66, 1, 62, 178, 15, 182, 96, 80,
    119, 23, 6, 166, 56, 99, 123, 138, 137, 21, 145, 185, 18, 70, 129, 95, 90, 149, 109, 124, 50,
    11, 152, 4, 31, 172, 40, 13, 32, 55, 159, 41, 8, 7, 144, 16, 26, 173, 81, 44, 103, 64, 9, 30,
    157, 126, 179, 148, 63, 188, 171, 106, 104, 158, 115, 34, 186, 29, 108, 53, 91, 169, 110, 27,
    59,
];

const MODE3_CARRIER_RANDOMIZE: &[usize] = &[
    62, 13, 371, 11, 285, 336, 365, 220, 226, 92, 56, 46, 120, 175, 298, 352, 172, 235, 53, 164,
    368, 187, 125, 82, 5, 45, 173, 258, 135, 182, 141, 273, 126, 264, 286, 88, 233, 61, 249, 367,
    310, 179, 155, 57, 123, 208, 14, 227, 100, 311, 205, 79, 184, 185, 328, 77, 115, 277, 112, 20,
    199, 178, 143, 152, 215, 204, 139, 234, 358, 192, 309, 183, 81, 129, 256, 314, 101, 43, 97,
    324, 142, 157, 90, 214, 102, 29, 303, 363, 261, 31, 22, 52, 305, 301, 293, 177, 116, 296, 85,
    196, 191, 114, 58, 198, 16, 167, 145, 119, 245, 113, 295, 193, 232, 17, 108, 283, 246, 64, 237,
    189, 128, 373, 302, 320, 239, 335, 356, 39, 347, 351, 73, 158, 276, 243, 99, 38, 287, 3, 330,
    153, 315, 117, 289, 213, 210, 149, 383, 337, 339, 151, 241, 321, 217, 30, 334, 161, 322, 49,
    176, 359, 12, 346, 60, 28, 229, 265, 288, 225, 382, 59, 181, 170, 319, 341, 86, 251, 133, 344,
    361, 109, 44, 369, 268, 257, 323, 55, 317, 381, 121, 360, 260, 275, 190, 19, 63, 18, 248, 9,
    240, 211, 150, 230, 332, 231, 71, 255, 350, 355, 83, 87, 154, 218, 138, 269, 348, 130, 160,
    278, 377, 216, 236, 308, 223, 254, 25, 98, 300, 201, 137, 219, 36, 325, 124, 66, 353, 169, 21,
    35, 107, 50, 106, 333, 326, 262, 252, 271, 263, 372, 136, 0, 366, 206, 159, 122, 188, 6, 284,
    96, 26, 200, 197, 186, 345, 340, 349, 103, 84, 228, 212, 2, 67, 318, 1, 74, 342, 166, 194, 33,
    68, 267, 111, 118, 140, 195, 105, 202, 291, 259, 23, 171, 65, 281, 24, 165, 8, 94, 222, 331,
    34, 238, 364, 376, 266, 89, 80, 253, 163, 280, 247, 4, 362, 379, 290, 279, 54, 78, 180, 72,
    316, 282, 131, 207, 343, 370, 306, 221, 132, 7, 148, 299, 168, 224, 48, 47, 357, 313, 75, 104,
    70, 147, 40, 110, 374, 69, 146, 37, 375, 354, 174, 41, 32, 304, 307, 312, 15, 272, 134, 242,
    203, 209, 380, 162, 297, 327, 10, 93, 42, 250, 156, 338, 292, 144, 378, 294, 329, 127, 270, 76,
    95, 91, 244, 274, 27, 51,
];

const NULL_PACKET: [u8; 188] = [
    0x47, 0x1f, 0xff, 0x10, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
];

#[allow(unused)]
#[derive(Clone, Copy, Debug)]
enum Mode {
    Mode1 = 1,
    Mode2 = 2,
    Mode3 = 3,
}
const PILOT_PRBS_INITIAL_STATE: u32 = 0b11111111111;

fn pilot_prbs(d: u32) -> u32 {
    let d11 = d & 1;
    let d9 = if (d & 0b100) == 0b100 { 1 } else { 0 };
    return (d >> 1) | ((d11 ^ d9) << 10);
}

fn encode_dbpsk(data: &[u8]) -> Vec<bool> {
    let mut encoded = Vec::with_capacity(data.len() * 8);
    let mut prev_b = false;
    encoded.push(prev_b);
    for byte in data {
        for bi in (0..8).rev() {
            let bit = (byte & (1 << bi)) != 0;
            prev_b ^= bit;
            encoded.push(prev_b);
        }
    }
    return encoded;
}

const OFDM_FRAME_SYMBOLS: usize = 204;

#[derive(Clone)]
struct Parameters {
    mode: Mode,
    guard_interval_ratio: usize,
}

#[derive(Clone)]
struct LayerParameters {
    modulation: CarrierModulation,
    segments: usize,
    coding_rate: CodingRate,
    time_interleave_length: usize,
}

fn tsp_worker(
    trans_params: Parameters,
    params: LayerParameters,
    receiver: std::sync::mpsc::Receiver<Vec<u8>>,
    sender: std::sync::mpsc::SyncSender<Vec<u8>>,
) {
    let mod_bits = params.modulation.get_bits();
    let mode = trans_params.mode;
    let rs_encoder = Encoder::new(TS_PARITY_SIZE);
    let tsp_per_frame =
        12 * mod_bits * (1 << (mode as usize - 1)) * params.segments * params.coding_rate.numer()
            / params.coding_rate.denom();
    let delay = tsp_per_frame - 11;
    let mut tsp_delay = VecDeque::new();
    for _ in 0..delay {
        let mut tsp = Vec::with_capacity(TSP_SIZE);
        let null_tsp = rs_encoder.encode(&NULL_PACKET);
        tsp.extend_from_slice(&null_tsp[1..]);
        tsp.push(TS_SYNC_BYTE);
        tsp_delay.push_back(tsp);
    }
    let mut byte_interleaver = ByteInterleaver::new();
    let mut prbs_state = BYTE_PRBS_INITIAL_STATE;
    let mut tsp_count = 0;
    while let Ok(packet) = receiver.recv() {
        let encoded_packet = rs_encoder.encode(&packet[0..TS_SIZE]);
        let mut tsp = Vec::with_capacity(TSP_SIZE);
        tsp.extend_from_slice(&encoded_packet[1..]);
        for b in &mut tsp {
            *b ^= next_byte_prbs(&mut prbs_state);
        }
        tsp.push(TS_SYNC_BYTE);
        next_byte_prbs(&mut prbs_state);
        tsp_delay.push_back(tsp);
        let delayed_tsp = tsp_delay.pop_front().unwrap();
        let mut ibuf = Vec::new();
        for b in delayed_tsp {
            ibuf.push(byte_interleaver.push(b));
        }
        sender.send(ibuf).unwrap();
        tsp_count += 1;
        if tsp_count >= tsp_per_frame {
            tsp_count = 0;
            prbs_state = BYTE_PRBS_INITIAL_STATE;
        }
    }
}

fn tsp_to_symbols_worker(
    trans_params: Parameters,
    params: LayerParameters,
    recv: std::sync::mpsc::Receiver<Vec<u8>>,
    sender: std::sync::mpsc::SyncSender<(Layer, Vec<Complex32>)>,
    layer: Layer,
) {
    let mode = trans_params.mode;
    let qpsk_mapping = [
        Complex32::from_polar(1.0f32, 45.0f32.to_radians()),
        Complex32::from_polar(1.0f32, 135.0f32.to_radians()),
        Complex32::from_polar(1.0f32, 225.0f32.to_radians()),
        Complex32::from_polar(1.0f32, 315.0f32.to_radians()),
    ];
    let mod_bits = params.modulation.get_bits();
    let mut bit_delay = VecDeque::from_iter(
        std::iter::repeat(false)
            .take((192 * (1 << (mode as usize - 1)) * params.segments - 120) * mod_bits),
    );
    let mut qpsk_delay = [
        VecDeque::from_iter(std::iter::repeat(false).take(0)),
        VecDeque::from_iter(std::iter::repeat(false).take(120)),
    ];
    let mut qam16_delay = [
        VecDeque::from_iter(std::iter::repeat(false).take(0)),
        VecDeque::from_iter(std::iter::repeat(false).take(40)),
        VecDeque::from_iter(std::iter::repeat(false).take(80)),
        VecDeque::from_iter(std::iter::repeat(false).take(120)),
    ];
    let mut qam64_delay = [
        VecDeque::from_iter(std::iter::repeat(false).take(0)),
        VecDeque::from_iter(std::iter::repeat(false).take(24)),
        VecDeque::from_iter(std::iter::repeat(false).take(48)),
        VecDeque::from_iter(std::iter::repeat(false).take(72)),
        VecDeque::from_iter(std::iter::repeat(false).take(96)),
        VecDeque::from_iter(std::iter::repeat(false).take(120)),
    ];
    let mut convolutional_encoder_state = 0;
    (_, convolutional_encoder_state, _) = convolve(
        &[TS_SYNC_BYTE],
        convolutional_encoder_state,
        &PUNCTURE_1_2,
        0,
    );
    let mut puncture_state = 0;
    let puncture_pattern = match params.coding_rate {
        CodingRate::Rate1_2 => &PUNCTURE_1_2[..],
        CodingRate::Rate2_3 => &PUNCTURE_2_3[..],
        CodingRate::Rate3_4 => &PUNCTURE_3_4[..],
        CodingRate::Rate5_6 => &PUNCTURE_5_6[..],
        CodingRate::Rate7_8 => &PUNCTURE_7_8[..],
    };
    let mut bit_buf = Vec::new();
    let mut layer_carriers = Vec::new();
    while let Ok(ibuf) = recv.recv() {
        let convolved;
        (convolved, convolutional_encoder_state, puncture_state) = convolve(
            &ibuf,
            convolutional_encoder_state,
            puncture_pattern,
            puncture_state,
        );
        for b in convolved {
            bit_delay.push_back(b);
            bit_buf.push(bit_delay.pop_front().unwrap());
        }
        let mut chunks = bit_buf.chunks_exact(mod_bits);
        while let Some(c) = chunks.next() {
            let symbol = match params.modulation {
                CarrierModulation::DQPSK => {
                    panic!("unsupported");
                }
                CarrierModulation::QPSK => {
                    qpsk_delay[0].push_back(c[0]);
                    qpsk_delay[1].push_back(c[1]);
                    let b0 = qpsk_delay[0].pop_front().unwrap();
                    let b1 = qpsk_delay[1].pop_front().unwrap();
                    qpsk_mapping[match (b0, b1) {
                        (false, false) => 0,
                        (true, false) => 1,
                        (true, true) => 2,
                        (false, true) => 3,
                    }]
                }
                CarrierModulation::QAM16 => {
                    qam16_delay[0].push_back(c[0]);
                    qam16_delay[1].push_back(c[1]);
                    qam16_delay[2].push_back(c[2]);
                    qam16_delay[3].push_back(c[3]);
                    let b0 = qam16_delay[0].pop_front().unwrap();
                    let b1 = qam16_delay[1].pop_front().unwrap();
                    let b2 = qam16_delay[2].pop_front().unwrap();
                    let b3 = qam16_delay[3].pop_front().unwrap();
                    Complex32::new(
                        match (b0, b2) {
                            (true, false) => -3,
                            (true, true) => -1,
                            (false, true) => 1,
                            (false, false) => 3,
                        } as f32,
                        match (b1, b3) {
                            (true, false) => -3,
                            (true, true) => -1,
                            (false, true) => 1,
                            (false, false) => 3,
                        } as f32,
                    ) / 10f32.sqrt()
                }
                CarrierModulation::QAM64 => {
                    qam64_delay[0].push_back(c[0]);
                    qam64_delay[1].push_back(c[1]);
                    qam64_delay[2].push_back(c[2]);
                    qam64_delay[3].push_back(c[3]);
                    qam64_delay[4].push_back(c[4]);
                    qam64_delay[5].push_back(c[5]);
                    let b0 = qam64_delay[0].pop_front().unwrap();
                    let b1 = qam64_delay[1].pop_front().unwrap();
                    let b2 = qam64_delay[2].pop_front().unwrap();
                    let b3 = qam64_delay[3].pop_front().unwrap();
                    let b4 = qam64_delay[4].pop_front().unwrap();
                    let b5 = qam64_delay[5].pop_front().unwrap();
                    Complex32::new(
                        match (b0, b2, b4) {
                            (true, false, false) => -7,
                            (true, false, true) => -5,
                            (true, true, true) => -3,
                            (true, true, false) => -1,
                            (false, true, false) => 1,
                            (false, true, true) => 3,
                            (false, false, true) => 5,
                            (false, false, false) => 7,
                        } as f32,
                        match (b1, b3, b5) {
                            (true, false, false) => -7,
                            (true, false, true) => -5,
                            (true, true, true) => -3,
                            (true, true, false) => -1,
                            (false, true, false) => 1,
                            (false, true, true) => 3,
                            (false, false, true) => 5,
                            (false, false, false) => 7,
                        } as f32,
                    ) / 42f32.sqrt()
                }
            };
            layer_carriers.push(symbol);
        }
        let size = layer_carriers.len();
        sender.send((layer, layer_carriers)).unwrap();
        layer_carriers = Vec::with_capacity(size);
        if !chunks.remainder().is_empty() {
            let remainder = chunks.remainder().to_vec();
            bit_buf.clear();
            bit_buf.extend_from_slice(&remainder);
        } else {
            bit_buf.clear();
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Layer {
    A,
    B,
    C,
}

fn simulate_model_receiver(
    params: &Parameters,
    layer_a: &LayerParameters,
    layer_b: &LayerParameters,
    layer_c: &LayerParameters,
) -> Vec<Option<Layer>> {
    let number_of_data_carriers = 96 * (1 << (params.mode as usize - 1));
    let symbol_len = 2048 * (1 << (params.mode as usize - 1));
    let guard_interval_len = symbol_len / params.guard_interval_ratio;
    let depunctured_tsp_bits = TSP_SIZE * 2 * 8;
    let tsp_read_interval = TSP_SIZE * 2;
    let mut ts_buffer = VecDeque::new();
    let mut clock = 0;
    let mut layer_a_buffer_bits = 0;
    let mut layer_b_buffer_bits = 0;
    let mut layer_c_buffer_bits = 0;
    let mut layer_a_depuncture_pos = 0;
    let mut layer_b_depuncture_pos = 0;
    let mut layer_c_depuncture_pos = 0;
    let mut ts_pattern = Vec::new();
    let tsp_delay = 3;
    for _ in 0..204 / (1 << (params.mode as usize - 1)) {
        for _ in 0..number_of_data_carriers * layer_a.segments {
            // layer_a_buffer_bits += 2
            //     * (((k + 1) * layer_a.modulation.get_bits() * layer_a.coding_rate.numer()
            //         / layer_a.coding_rate.denom())
            //         - (k * layer_a.modulation.get_bits() * layer_a.coding_rate.numer()
            //             / layer_a.coding_rate.denom()));
            for _ in 0..layer_a.modulation.get_bits() {
                layer_a_buffer_bits += 1;
                if layer_a_depuncture_pos >= 2 {
                    layer_a_buffer_bits += 1;
                }
                layer_a_depuncture_pos = (layer_a_depuncture_pos + 1) % layer_a.coding_rate.denom();
                if layer_a_buffer_bits >= depunctured_tsp_bits {
                    layer_a_buffer_bits -= depunctured_tsp_bits;
                    ts_buffer.push_back(Layer::A);
                }
            }
            if (clock % tsp_read_interval) == 0 {
                if clock >= tsp_read_interval * tsp_delay {
                    ts_pattern.push(ts_buffer.pop_front());
                }
            }
            clock += 1;
        }
        for _ in 0..number_of_data_carriers * layer_b.segments {
            // layer_b_buffer_bits += 2
            //     * (((k + 1) * layer_b.modulation.get_bits() * layer_b.coding_rate.numer()
            //         / layer_b.coding_rate.denom())
            //         - (k * layer_b.modulation.get_bits() * layer_b.coding_rate.numer()
            //             / layer_b.coding_rate.denom()));
            for _ in 0..layer_b.modulation.get_bits() {
                layer_b_buffer_bits += 1;
                if layer_b_depuncture_pos >= 2 {
                    layer_b_buffer_bits += 1;
                }
                layer_b_depuncture_pos = (layer_b_depuncture_pos + 1) % layer_b.coding_rate.denom();
                if layer_b_buffer_bits >= depunctured_tsp_bits {
                    layer_b_buffer_bits -= depunctured_tsp_bits;
                    ts_buffer.push_back(Layer::B);
                }
            }
            if (clock % tsp_read_interval) == 0 {
                if clock >= tsp_read_interval * tsp_delay {
                    ts_pattern.push(ts_buffer.pop_front());
                }
            }
            clock += 1;
        }
        for _ in 0..number_of_data_carriers * layer_c.segments {
            // layer_c_buffer_bits += 2
            //     * (((k + 1) * layer_c.modulation.get_bits() * layer_c.coding_rate.numer()
            //         / layer_c.coding_rate.denom())
            //         - (k * layer_c.modulation.get_bits() * layer_c.coding_rate.numer()
            //             / layer_c.coding_rate.denom()));
            for _ in 0..layer_c.modulation.get_bits() {
                layer_c_buffer_bits += 1;
                if layer_c_depuncture_pos >= 2 {
                    layer_c_buffer_bits += 1;
                }
                layer_c_depuncture_pos = (layer_c_depuncture_pos + 1) % layer_c.coding_rate.denom();
                if layer_c_buffer_bits >= depunctured_tsp_bits {
                    layer_c_buffer_bits -= depunctured_tsp_bits;
                    ts_buffer.push_back(Layer::C);
                }
            }
            if (clock % tsp_read_interval) == 0 {
                if clock >= tsp_read_interval * tsp_delay {
                    ts_pattern.push(ts_buffer.pop_front());
                }
            }
            clock += 1;
        }
        for _ in number_of_data_carriers * SEGMENTS..symbol_len + guard_interval_len {
            if (clock % tsp_read_interval) == 0 {
                if clock >= tsp_read_interval * tsp_delay {
                    ts_pattern.push(ts_buffer.pop_front());
                }
            }
            clock += 1;
        }
    }
    for _ in 0..tsp_delay {
        ts_pattern.push(ts_buffer.pop_front());
    }
    assert_eq!(layer_a_buffer_bits, 0);
    assert_eq!(layer_b_buffer_bits, 0);
    assert_eq!(layer_c_buffer_bits, 0);
    assert_eq!(ts_buffer.len(), 0);
    for _ in 1..params.mode as usize {
        ts_pattern.extend_from_within(..);
    }
    return ts_pattern;
}

fn ts_worker(
    params: &Parameters,
    layer_a_params: &LayerParameters,
    layer_a_ts_sender: std::sync::mpsc::SyncSender<Vec<u8>>,
    layer_b_params: &LayerParameters,
    layer_b_ts_sender: std::sync::mpsc::SyncSender<Vec<u8>>,
    layer_c_params: &LayerParameters,
    layer_c_ts_sender: std::sync::mpsc::SyncSender<Vec<u8>>,
) {
    let frame_pattern =
        simulate_model_receiver(&params, &layer_a_params, &layer_b_params, &layer_c_params);
    let mut stdin = std::io::stdin();
    let mut frame_buffer = Vec::with_capacity(frame_pattern.len() * TS_SIZE);
    frame_buffer.resize(frame_pattern.len() * TS_SIZE, 0);
    loop {
        stdin.read_exact(&mut frame_buffer).unwrap();
        for (i, l) in frame_pattern.iter().enumerate() {
            if let Some(l) = l {
                let packet = frame_buffer[i * TS_SIZE..(i + 1) * TS_SIZE].to_vec();
                match l {
                    Layer::A => layer_a_ts_sender.send(packet).unwrap(),
                    Layer::B => layer_b_ts_sender.send(packet).unwrap(),
                    Layer::C => layer_c_ts_sender.send(packet).unwrap(),
                }
            }
        }
    }
}

fn main() -> io::Result<()> {
    let layer_a_params = LayerParameters {
        modulation: CarrierModulation::QPSK,
        segments: 1,
        coding_rate: CodingRate::Rate2_3,
        time_interleave_length: 4,
    };
    let layer_b_params = LayerParameters {
        modulation: CarrierModulation::QAM64,
        segments: 12,
        coding_rate: CodingRate::Rate3_4,
        time_interleave_length: 2,
    };
    let layer_c_params = LayerParameters {
        modulation: CarrierModulation::QAM64,
        segments: 0,
        coding_rate: CodingRate::Rate7_8,
        time_interleave_length: 4,
    };
    let partial_reception = true;
    let mode = Mode::Mode3;
    let guard_interval_ratio = 8;
    let params = Parameters {
        mode,
        guard_interval_ratio,
    };
    let tmcc = TMCC {
        segment_type: SegmentType::Coherent,
        system_idenfication: SystemIdentification::Television,
        switcihg_indicator: 0b1111,
        startup_control: false,
        current: TMCCTransmissionParameters {
            partial_reception,
            layer_a: TMCCLayerParameters::from(&layer_a_params, params.mode),
            layer_b: TMCCLayerParameters::from(&layer_b_params, params.mode),
            layer_c: TMCCLayerParameters::from(&layer_c_params, params.mode),
        },
        next: TMCCTransmissionParameters {
            partial_reception,
            layer_a: TMCCLayerParameters::from(&layer_a_params, params.mode),
            layer_b: TMCCLayerParameters::from(&layer_b_params, params.mode),
            layer_c: TMCCLayerParameters::from(&layer_c_params, params.mode),
        },
    };
    let args: Vec<_> = std::env::args().collect();
    let name = args[1].to_string();
    let mut output: Box<dyn Write> = if name == "-" {
        Box::new(std::io::stdout())
    } else {
        Box::new(
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(name)?,
        )
    };
    let number_of_data_carriers = 96 * (1 << (mode as usize - 1));
    let number_of_carriers = 108 * (1 << (mode as usize - 1));
    let symbol_len = 2048 * (1 << (mode as usize - 1));
    let guard_interval_len = symbol_len / guard_interval_ratio;
    let mut time_interleavers_layer_a = Vec::with_capacity(layer_a_params.segments);
    time_interleavers_layer_a.resize_with(layer_a_params.segments, || {
        TimeInterleaver::new(
            layer_a_params.time_interleave_length,
            number_of_data_carriers,
            OFDM_FRAME_SYMBOLS - (95 * layer_a_params.time_interleave_length) % OFDM_FRAME_SYMBOLS,
        )
    });
    let mut time_interleavers_layer_b = Vec::with_capacity(layer_b_params.segments);
    time_interleavers_layer_b.resize_with(layer_b_params.segments, || {
        TimeInterleaver::new(
            layer_b_params.time_interleave_length,
            number_of_data_carriers,
            OFDM_FRAME_SYMBOLS - (95 * layer_b_params.time_interleave_length) % OFDM_FRAME_SYMBOLS,
        )
    });
    let mut time_interleavers_layer_c = Vec::with_capacity(layer_c_params.segments);
    time_interleavers_layer_c.resize_with(layer_c_params.segments, || {
        TimeInterleaver::new(
            layer_c_params.time_interleave_length,
            number_of_data_carriers,
            OFDM_FRAME_SYMBOLS - (95 * layer_c_params.time_interleave_length) % OFDM_FRAME_SYMBOLS,
        )
    });
    let mut ofdm_symbol_index = OFDM_FRAME_SYMBOLS - 2;
    let mut ofdm_frame_index = 0;
    let mut ac_data = Vec::with_capacity((OFDM_FRAME_SYMBOLS + 7) / 8);
    ac_data.resize((OFDM_FRAME_SYMBOLS + 7) / 8, 0xffu8);
    let mut tmcc_data = Vec::from_iter(tmcc.to_bytes());
    tmcc_data.resize((OFDM_FRAME_SYMBOLS + 7) / 8, 0x00);
    let encoded_ac = encode_dbpsk(&ac_data);
    tmcc_data[0] = 0b00110101;
    tmcc_data[1] = 0b11101110;
    let encoded_tmcc_even = encode_dbpsk(&tmcc_data);
    tmcc_data[0] = 0b11001010;
    tmcc_data[1] = 0b00010001;
    let encoded_tmcc_odd = encode_dbpsk(&tmcc_data);
    let mut prbs_table = Vec::<bool>::new();
    let mut pilot_prbs_state = PILOT_PRBS_INITIAL_STATE;
    for _ in 0..number_of_carriers * SEGMENTS + 1 {
        prbs_table.push((pilot_prbs_state & 1) == 1);
        pilot_prbs_state = pilot_prbs(pilot_prbs_state);
    }
    let coherent_modulation_segments = if !partial_reception {
        layer_a_params.segments
    } else {
        0
    } + layer_b_params.segments
        + layer_c_params.segments;
    let partial_reception_segments = if partial_reception { 1 } else { 0 };
    let mut time_interleaved_partial_reception_carriers =
        Vec::with_capacity(partial_reception_segments * number_of_data_carriers);
    time_interleaved_partial_reception_carriers.resize(
        partial_reception_segments * number_of_data_carriers,
        Complex32::ZERO,
    );
    let mut time_interleaved_coherent_carriers =
        Vec::with_capacity(coherent_modulation_segments * number_of_data_carriers);
    time_interleaved_coherent_carriers.resize(
        coherent_modulation_segments * number_of_data_carriers,
        Complex32::ZERO,
    );
    let mut inter_segment_interleaved_layer_carriers =
        Vec::with_capacity(coherent_modulation_segments * number_of_data_carriers);
    inter_segment_interleaved_layer_carriers.resize(
        coherent_modulation_segments * number_of_data_carriers,
        Complex32::ZERO,
    );
    let mut intra_segment_interleaved_layer_carriers =
        Vec::with_capacity(coherent_modulation_segments * number_of_data_carriers);
    intra_segment_interleaved_layer_carriers.resize(
        coherent_modulation_segments * number_of_data_carriers,
        Complex32::ZERO,
    );
    let mut randomized_layer_carriers =
        Vec::with_capacity(coherent_modulation_segments * number_of_data_carriers);
    randomized_layer_carriers.resize(
        coherent_modulation_segments * number_of_data_carriers,
        Complex32::ZERO,
    );
    let mut randomized_partial_reception_carriers =
        Vec::with_capacity(partial_reception_segments * number_of_data_carriers);
    randomized_partial_reception_carriers.resize(
        partial_reception_segments * number_of_data_carriers,
        Complex32::ZERO,
    );
    let carrier_randomize = match mode {
        Mode::Mode1 => &MODE1_CARRIER_RANDOMIZE[..],
        Mode::Mode2 => &MODE2_CARRIER_RANDOMIZE[..],
        Mode::Mode3 => &MODE3_CARRIER_RANDOMIZE[..],
    };
    let mut frame_begin_time = Instant::now();
    let (layer_a_ts_sender, layer_a_ts_receiver) = std::sync::mpsc::sync_channel(65536);
    let (layer_b_ts_sender, layer_b_ts_receiver) = std::sync::mpsc::sync_channel(65536);
    let (layer_c_ts_sender, layer_c_ts_receiver) = std::sync::mpsc::sync_channel(65536);
    let p = params.clone();
    let p1 = layer_a_params.clone();
    let p2 = layer_b_params.clone();
    let p3 = layer_c_params.clone();
    std::thread::spawn(move || {
        ts_worker(
            &p,
            &p1,
            layer_a_ts_sender,
            &p2,
            layer_b_ts_sender,
            &p3,
            layer_c_ts_sender,
        );
    });
    let (layer_a_interleaved_tsp_sender, layer_a_interleaved_tsp_receiver) =
        std::sync::mpsc::sync_channel(65536);
    let (layer_b_interleaved_tsp_sender, layer_b_interleaved_tsp_receiver) =
        std::sync::mpsc::sync_channel(65536);
    let (layer_c_interleaved_tsp_sender, layer_c_interleaved_tsp_receiver) =
        std::sync::mpsc::sync_channel(65536);
    let p = layer_a_params.clone();
    std::thread::spawn(move || {
        tsp_worker(
            Parameters {
                mode,
                guard_interval_ratio,
            },
            p,
            layer_a_ts_receiver,
            layer_a_interleaved_tsp_sender,
        )
    });
    if layer_b_params.segments > 0 {
        let p = layer_b_params.clone();
        std::thread::spawn(move || {
            tsp_worker(
                Parameters {
                    mode,
                    guard_interval_ratio,
                },
                p,
                layer_b_ts_receiver,
                layer_b_interleaved_tsp_sender,
            )
        });
    }
    if layer_c_params.segments > 0 {
        let p = layer_c_params.clone();
        std::thread::spawn(move || {
            tsp_worker(
                Parameters {
                    mode,
                    guard_interval_ratio,
                },
                p,
                layer_c_ts_receiver,
                layer_c_interleaved_tsp_sender,
            )
        });
    }
    let (layer_a_symbol_sender, symbol_receiver) = std::sync::mpsc::sync_channel(65536);
    let layer_b_symbol_sender = layer_a_symbol_sender.clone();
    let layer_c_symbol_sender = layer_a_symbol_sender.clone();
    let p = layer_a_params.clone();
    std::thread::spawn(move || {
        tsp_to_symbols_worker(
            Parameters {
                mode,
                guard_interval_ratio,
            },
            p,
            layer_a_interleaved_tsp_receiver,
            layer_a_symbol_sender,
            Layer::A,
        )
    });
    if layer_b_params.segments > 0 {
        let p = layer_b_params.clone();
        std::thread::spawn(move || {
            tsp_to_symbols_worker(
                Parameters {
                    mode,
                    guard_interval_ratio,
                },
                p,
                layer_b_interleaved_tsp_receiver,
                layer_b_symbol_sender,
                Layer::B,
            )
        });
    }
    if layer_c_params.segments > 0 {
        let p = layer_c_params.clone();
        std::thread::spawn(move || {
            tsp_to_symbols_worker(
                Parameters {
                    mode,
                    guard_interval_ratio,
                },
                p,
                layer_c_interleaved_tsp_receiver,
                layer_c_symbol_sender,
                Layer::C,
            )
        });
    }
    let mut layer_a_symbol_buffer = VecDeque::new();
    let mut layer_b_symbol_buffer = VecDeque::new();
    let mut layer_c_symbol_buffer = VecDeque::new();
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(symbol_len);
    loop {
        if let Ok((layer, carriers)) = symbol_receiver.recv() {
            match layer {
                Layer::A => {
                    layer_a_symbol_buffer.extend(carriers);
                }
                Layer::B => {
                    layer_b_symbol_buffer.extend(carriers);
                }
                Layer::C => {
                    layer_c_symbol_buffer.extend(carriers);
                }
            }
        } else {
            break;
        }
        if layer_a_symbol_buffer.len() >= number_of_data_carriers * layer_a_params.segments
            && layer_b_symbol_buffer.len() >= number_of_data_carriers * layer_b_params.segments
            && layer_c_symbol_buffer.len() >= number_of_data_carriers * layer_c_params.segments
        {
            for segment in 0..layer_a_params.segments {
                let time_interleaver = &mut time_interleavers_layer_a[segment];
                for i in 0..number_of_data_carriers {
                    if partial_reception {
                        time_interleaved_partial_reception_carriers
                            [segment * number_of_data_carriers + i] =
                            time_interleaver.push(layer_a_symbol_buffer.pop_front().unwrap(), i);
                    } else {
                        time_interleaved_coherent_carriers[segment * number_of_data_carriers + i] =
                            time_interleaver.push(layer_a_symbol_buffer.pop_front().unwrap(), i);
                    }
                }
            }
            let offset = if !partial_reception {
                layer_a_params.segments * number_of_data_carriers
            } else {
                0
            };
            for segment in 0..layer_b_params.segments {
                let time_interleaver = &mut time_interleavers_layer_b[segment];
                for i in 0..number_of_data_carriers {
                    time_interleaved_coherent_carriers
                        [offset + segment * number_of_data_carriers + i] =
                        time_interleaver.push(layer_b_symbol_buffer.pop_front().unwrap(), i);
                }
            }
            let offset = offset + layer_b_params.segments * number_of_data_carriers;
            for segment in 0..layer_c_params.segments {
                let time_interleaver = &mut time_interleavers_layer_c[segment];
                for i in 0..number_of_data_carriers {
                    time_interleaved_coherent_carriers
                        [offset + segment * number_of_data_carriers + i] =
                        time_interleaver.push(layer_c_symbol_buffer.pop_front().unwrap(), i);
                }
            }
            for segment in 0..coherent_modulation_segments {
                for i in 0..number_of_data_carriers {
                    inter_segment_interleaved_layer_carriers
                        [segment * number_of_data_carriers + i] =
                        time_interleaved_coherent_carriers
                            [segment + i * coherent_modulation_segments];
                }
            }
            for segment in 0..coherent_modulation_segments {
                let dest_segment = &mut intra_segment_interleaved_layer_carriers
                    [segment * number_of_data_carriers..(segment + 1) * number_of_data_carriers];
                let source_segment = &inter_segment_interleaved_layer_carriers
                    [segment * number_of_data_carriers..(segment + 1) * number_of_data_carriers];
                for i in 0..number_of_data_carriers {
                    dest_segment[i] = source_segment
                        [(i + segment + partial_reception_segments) % number_of_data_carriers];
                }
            }
            for segment in 0..coherent_modulation_segments {
                for i in 0..number_of_data_carriers {
                    randomized_layer_carriers
                        [carrier_randomize[i] + number_of_data_carriers * segment] =
                        intra_segment_interleaved_layer_carriers
                            [i + number_of_data_carriers * segment];
                }
            }
            for segment in 0..partial_reception_segments {
                for i in 0..number_of_data_carriers {
                    randomized_partial_reception_carriers
                        [carrier_randomize[i] + number_of_data_carriers * segment] =
                        time_interleaved_partial_reception_carriers
                            [i + number_of_data_carriers * segment];
                }
            }
            let mut carriers = Vec::with_capacity(symbol_len);
            carriers.resize(symbol_len, Complex32::ZERO);
            let seg_off = symbol_len / 2 - number_of_carriers / 2 - 6 * number_of_carriers;
            let segment_offsets = [6, 5, 7, 4, 8, 3, 9, 2, 10, 1, 11, 0, 12];
            let mut data_index = 0;
            for segment in 0..partial_reception_segments + coherent_modulation_segments {
                let tmcc_carriers = TMCC_CARRIER[segment_offsets[segment]];
                let ac_carriers = AC_CARRIER[segment_offsets[segment]];
                let segment_carrier_index = number_of_carriers * segment_offsets[segment];
                let segment_carries = &mut carriers[seg_off + segment_carrier_index
                    ..seg_off + segment_carrier_index + number_of_carriers];
                for i in 0..number_of_carriers {
                    if i % 12 == ofdm_symbol_index % 4 * 3 {
                        match prbs_table[segment_carrier_index + i] {
                            true => {
                                segment_carries[i] = Complex32::new(-4f32 / 3f32, 0f32);
                            }
                            false => {
                                segment_carries[i] = Complex32::new(4f32 / 3f32, 0f32);
                            }
                        }
                    } else if tmcc_carriers.contains(&i) {
                        match if ofdm_frame_index % 2 == 0 {
                            encoded_tmcc_even[ofdm_symbol_index]
                        } else {
                            encoded_tmcc_odd[ofdm_symbol_index]
                        } ^ prbs_table[segment_carrier_index + i]
                        {
                            true => {
                                segment_carries[i] = Complex32::new(-4f32 / 3f32, 0f32);
                            }
                            false => {
                                segment_carries[i] = Complex32::new(4f32 / 3f32, 0f32);
                            }
                        }
                    } else if ac_carriers.contains(&i) {
                        match encoded_ac[ofdm_symbol_index] ^ prbs_table[segment_carrier_index + i]
                        {
                            true => {
                                segment_carries[i] = Complex32::new(-4f32 / 3f32, 0f32);
                            }
                            false => {
                                segment_carries[i] = Complex32::new(4f32 / 3f32, 0f32);
                            }
                        }
                    } else {
                        if segment < partial_reception_segments {
                            segment_carries[i] = randomized_partial_reception_carriers[data_index];
                        } else {
                            segment_carries[i] = randomized_layer_carriers
                                [data_index - partial_reception_segments * number_of_data_carriers];
                        }
                        data_index += 1;
                    }
                }
            }
            assert_eq!(
                data_index,
                randomized_partial_reception_carriers.len() + randomized_layer_carriers.len()
            );
            carriers[seg_off + number_of_carriers * SEGMENTS] =
                match (prbs_table[number_of_carriers * SEGMENTS], mode) {
                    (true, Mode::Mode1) => Complex32::new(-4f32 / 3f32, 0f32),
                    (false, Mode::Mode2 | Mode::Mode3) => Complex32::new(4f32 / 3f32, 0f32),
                    _ => panic!(),
                };
            let (negative_freq, positive_freq) = carriers.split_at_mut(symbol_len / 2);
            positive_freq.swap_with_slice(negative_freq);
            ifft.process(&mut carriers);
            let guard_interval = &carriers[carriers.len() - guard_interval_len..];
            if ofdm_frame_index > 2 {
                let mut buf = Vec::new();
                for c in guard_interval {
                    buf.extend_from_slice(&c.re.to_le_bytes());
                    buf.extend_from_slice(&c.im.to_le_bytes());
                }
                for c in &carriers {
                    buf.extend_from_slice(&c.re.to_le_bytes());
                    buf.extend_from_slice(&c.im.to_le_bytes());
                }
                output.write_all(&buf)?;
            }
            ofdm_symbol_index += 1;
            if ofdm_symbol_index == OFDM_FRAME_SYMBOLS {
                ofdm_symbol_index = 0;
                eprintln!(
                    "{ofdm_frame_index} {:.03}",
                    (Instant::now() - frame_begin_time).as_secs_f32()
                );
                frame_begin_time = Instant::now();
                ofdm_frame_index += 1;
            }
        }
    }
    return Ok(());
}

#[allow(unused)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum SegmentType {
    Coherent = 0,
    Undefined1 = 1,
    Undefined2 = 2,
    Undefined3 = 3,
    Undefined4 = 4,
    Undefined5 = 5,
    Undefined6 = 6,
    Differential = 7,
}

#[allow(unused)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum SystemIdentification {
    Television = 0,
    Sound = 1,
    Undefined2 = 2,
    Undefined3 = 3,
}

#[allow(unused)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum CarrierModulation {
    DQPSK,
    QPSK,
    QAM16,
    QAM64,
}

impl CarrierModulation {
    fn get_bits(&self) -> usize {
        return match self {
            CarrierModulation::DQPSK | CarrierModulation::QPSK => 2,
            CarrierModulation::QAM16 => 4,
            CarrierModulation::QAM64 => 6,
        };
    }
}

#[allow(unused)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum TMCCCarrierModulation {
    DQPSK = 0,
    QPSK = 1,
    QAM16 = 2,
    QAM64 = 3,
    Undefined4 = 4,
    Undefined5 = 5,
    Undefined6 = 6,
    Unused = 7,
}

impl From<CarrierModulation> for TMCCCarrierModulation {
    fn from(value: CarrierModulation) -> Self {
        return match value {
            CarrierModulation::DQPSK => TMCCCarrierModulation::DQPSK,
            CarrierModulation::QPSK => TMCCCarrierModulation::QPSK,
            CarrierModulation::QAM16 => TMCCCarrierModulation::QAM16,
            CarrierModulation::QAM64 => TMCCCarrierModulation::QAM64,
        };
    }
}

#[allow(unused)]
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
enum CodingRate {
    Rate1_2,
    Rate2_3,
    Rate3_4,
    Rate5_6,
    Rate7_8,
}

impl CodingRate {
    pub fn numer(&self) -> usize {
        return match self {
            CodingRate::Rate1_2 => 1,
            CodingRate::Rate2_3 => 2,
            CodingRate::Rate3_4 => 3,
            CodingRate::Rate5_6 => 5,
            CodingRate::Rate7_8 => 7,
        };
    }
    pub fn denom(&self) -> usize {
        return match self {
            CodingRate::Rate1_2 => 2,
            CodingRate::Rate2_3 => 3,
            CodingRate::Rate3_4 => 4,
            CodingRate::Rate5_6 => 6,
            CodingRate::Rate7_8 => 8,
        };
    }
}

#[allow(unused)]
#[derive(Debug, Clone, PartialEq, Eq, Copy)]
enum TMCCCodingRate {
    Rate1_2 = 0,
    Rate2_3 = 1,
    Rate3_4 = 2,
    Rate5_6 = 3,
    Rate7_8 = 4,
    Undefined5 = 5,
    Undefined6 = 6,
    Unused = 7,
}

impl From<CodingRate> for TMCCCodingRate {
    fn from(value: CodingRate) -> Self {
        return match value {
            CodingRate::Rate1_2 => TMCCCodingRate::Rate1_2,
            CodingRate::Rate2_3 => TMCCCodingRate::Rate2_3,
            CodingRate::Rate3_4 => TMCCCodingRate::Rate3_4,
            CodingRate::Rate5_6 => TMCCCodingRate::Rate5_6,
            CodingRate::Rate7_8 => TMCCCodingRate::Rate7_8,
        };
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TMCCLayerParameters {
    carrier_modulation: TMCCCarrierModulation,
    coding_rate: TMCCCodingRate,
    time_interleaving_length: u8,
    number_of_segments: u8,
}

impl TMCCLayerParameters {
    fn from(value: &LayerParameters, mode: Mode) -> Self {
        if value.segments == 0 {
            return TMCCLayerParameters {
                carrier_modulation: TMCCCarrierModulation::Unused,
                coding_rate: TMCCCodingRate::Unused,
                number_of_segments: 0b1111,
                time_interleaving_length: 0b111,
            };
        }
        return TMCCLayerParameters {
            carrier_modulation: TMCCCarrierModulation::from(value.modulation),
            coding_rate: TMCCCodingRate::from(value.coding_rate),
            number_of_segments: value.segments as u8,
            time_interleaving_length: match (value.time_interleave_length, mode) {
                (0, _) => 0b000,
                (4, Mode::Mode1) => 0b001,
                (2, Mode::Mode2) => 0b001,
                (1, Mode::Mode3) => 0b001,
                (8, Mode::Mode1) => 0b010,
                (4, Mode::Mode2) => 0b010,
                (2, Mode::Mode3) => 0b010,
                (16, Mode::Mode1) => 0b011,
                (8, Mode::Mode2) => 0b011,
                (4, Mode::Mode3) => 0b011,
                _ => 0b111,
            },
        };
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TMCCTransmissionParameters {
    partial_reception: bool,
    layer_a: TMCCLayerParameters,
    layer_b: TMCCLayerParameters,
    layer_c: TMCCLayerParameters,
}

impl TMCCTransmissionParameters {
    fn to_bytes(&self) -> [u8; 5] {
        let mut buf = [0u8; 5];
        let mut pos = 0;
        set_bits(&mut buf, pos, 1, if self.partial_reception { 1 } else { 0 });
        pos += 1;
        set_bits(&mut buf, pos, 3, self.layer_a.carrier_modulation as u32);
        pos += 3;
        set_bits(&mut buf, pos, 3, self.layer_a.coding_rate as u32);
        pos += 3;
        set_bits(
            &mut buf,
            pos,
            3,
            self.layer_a.time_interleaving_length as u32,
        );
        pos += 3;
        set_bits(&mut buf, pos, 4, self.layer_a.number_of_segments as u32);
        pos += 4;
        set_bits(&mut buf, pos, 3, self.layer_b.carrier_modulation as u32);
        pos += 3;
        set_bits(&mut buf, pos, 3, self.layer_b.coding_rate as u32);
        pos += 3;
        set_bits(
            &mut buf,
            pos,
            3,
            self.layer_b.time_interleaving_length as u32,
        );
        pos += 3;
        set_bits(&mut buf, pos, 4, self.layer_b.number_of_segments as u32);
        pos += 4;
        set_bits(&mut buf, pos, 3, self.layer_c.carrier_modulation as u32);
        pos += 3;
        set_bits(&mut buf, pos, 3, self.layer_c.coding_rate as u32);
        pos += 3;
        set_bits(
            &mut buf,
            pos,
            3,
            self.layer_c.time_interleaving_length as u32,
        );
        pos += 3;
        set_bits(&mut buf, pos, 4, self.layer_c.number_of_segments as u32);
        pos += 4;
        assert_eq!(pos, 40);
        return buf;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TMCC {
    segment_type: SegmentType,
    system_idenfication: SystemIdentification,
    switcihg_indicator: u8,
    startup_control: bool,
    current: TMCCTransmissionParameters,
    next: TMCCTransmissionParameters,
}

impl TMCC {
    fn to_bytes(&self) -> [u8; 26] {
        let mut buf = [0u8; 26];
        let mut pos = 16;
        set_bits(&mut buf, pos, 3, self.segment_type as u32);
        pos += 3;
        set_bits(&mut buf, pos, 2, self.system_idenfication as u32);
        pos += 2;
        set_bits(&mut buf, pos, 4, self.switcihg_indicator as u32);
        pos += 4;
        set_bits(&mut buf, pos, 1, if self.startup_control { 1 } else { 0 });
        pos += 1;
        for b in self.current.to_bytes() {
            set_bits(&mut buf, pos, 8, b as u32);
            pos += 8;
        }
        for b in self.next.to_bytes() {
            set_bits(&mut buf, pos, 8, b as u32);
            pos += 8;
        }
        set_bits(&mut buf, pos, 3, 0b111);
        pos += 3;
        set_bits(&mut buf, pos, 12, 0b111111111111);
        pos += 12;
        assert_eq!(pos, 121);
        let parity = calc_272_190_parity(&buf, 19, 120);
        set_bits(&mut buf, pos, 82 - 64, (parity >> 64) as u32);
        pos += 82 - 64;
        set_bits(&mut buf, pos, 32, (parity >> 32) as u32);
        pos += 32;
        set_bits(&mut buf, pos, 32, parity as u32);
        pos += 32;
        assert_eq!(pos, 203);
        return buf;
    }
}

fn set_bits(buf: &mut [u8], mut pos: usize, bits: usize, value: u32) {
    for i in (0..bits).rev() {
        let bit = ((value >> i) & 1) as u8;
        buf[pos / 8] &= !(1 << (7 - pos % 8));
        buf[pos / 8] |= bit << (7 - pos % 8);
        pos += 1;
    }
}

fn calc_272_190_parity(data: &[u8], begin: usize, end: usize) -> u128 {
    const POLY: u128 = (1 << 77)
        | (1 << 76)
        | (1 << 71)
        | (1 << 67)
        | (1 << 66)
        | (1 << 56)
        | (1 << 52)
        | (1 << 48)
        | (1 << 40)
        | (1 << 36)
        | (1 << 34)
        | (1 << 24)
        | (1 << 22)
        | (1 << 18)
        | (1 << 10)
        | (1 << 4)
        | 1;
    let mut d = 0u128;
    for in_bi in begin..=end {
        d <<= 1;
        if (data[in_bi / 8] & (1 << (7 - in_bi % 8)) != 0) ^ ((d & (1 << 82)) != 0) {
            d ^= POLY;
        }
    }
    return d & ((1 << 82) - 1);
}
