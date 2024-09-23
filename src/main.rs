use reed_solomon::Encoder;
use rustfft::{
    num_complex::Complex32,
    FftPlanner,
};
use std::{
    collections::VecDeque,
    io::{self, Read, Write},
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
    index: usize,
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
        return Self { buffers, index: 0 };
    }
    fn push(&mut self, symbol: Complex32) -> Complex32 {
        let len = self.buffers.len();
        let buffer = &mut self.buffers[self.index];
        buffer.push_back(symbol);
        self.index = (self.index + 1) % len;
        return buffer.pop_front().unwrap();
    }
}

fn convolve(data: &[u8], mut d: u8) -> (Vec<bool>, u8) {
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
            v.push(c0);
            v.push(c1);
            d >>= 1;
        }
    }
    return (v, d);
}

fn puncture(data: &[bool]) -> Vec<bool> {
    let mut v = Vec::with_capacity(data.len() * 2 / 3);
    for c in data.chunks(4) {
        let x1 = c[0];
        let y1 = c[1];
        let y2 = c[3];
        v.push(x1);
        v.push(y1);
        v.push(y2);
    }
    return v;
}

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
fn main() -> io::Result<()> {
    let qpsk_mapping = [
        Complex32::from_polar(1.0f32, 45.0f32.to_radians()),
        Complex32::from_polar(1.0f32, 135.0f32.to_radians()),
        Complex32::from_polar(1.0f32, 225.0f32.to_radians()),
        Complex32::from_polar(1.0f32, 315.0f32.to_radians()),
    ];
    let args: Vec<_> = std::env::args().collect();
    let mut output = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(args[1].to_string())?;
    let mut stdin = std::io::stdin();
    let mode = Mode::Mode3;
    let number_of_data_carriers = 96 * (1 << (mode as usize - 1));
    let number_of_carriers = 108 * (1 << (mode as usize - 1));
    let guard_interval_ratio = 8;
    let symbol_len = 2048 * (1 << (mode as usize - 1));
    let guard_interval_len = symbol_len / guard_interval_ratio;
    let segments = 1;
    let mut bit_delay = VecDeque::from_iter(std::iter::repeat(false).take(1536 * segments - 240));
    let mut qpsk_delay = VecDeque::from_iter(std::iter::repeat(false).take(120));
    let time_interleave_length = 4;
    let mut symbol_delay = Vec::with_capacity(number_of_data_carriers);
    symbol_delay.resize_with(number_of_data_carriers, || {
        VecDeque::from_iter(std::iter::repeat(Complex32::ZERO).take(match mode {
            Mode::Mode1 | Mode::Mode2 => 7 * time_interleave_length,
            Mode::Mode3 => (109 * time_interleave_length) % 204,
        }))
    });
    let mut time_interleaver = TimeInterleaver::new(
        time_interleave_length,
        number_of_data_carriers,
        match mode {
            Mode::Mode1 | Mode::Mode2 => 7 * time_interleave_length,
            Mode::Mode3 => 204 - (95 * time_interleave_length) % 204,
        },
    );
    let mut byte_interleaver = ByteInterleaver::new();
    let rs_encoder = Encoder::new(TS_PARITY_SIZE);
    let mut ofdm_symbol_index = OFDM_FRAME_SYMBOLS - 2;
    let mut ofdm_frame_index = 0;
    let delay = 64 * 1 - 11;
    let mut tsp_delay = VecDeque::new();
    for _ in 0..delay {
        let mut tsp = Vec::with_capacity(TSP_SIZE);
        let null_tsp = rs_encoder.encode(&NULL_PACKET);
        tsp.extend_from_slice(&null_tsp[1..]);
        tsp.push(TS_SYNC_BYTE);
        tsp_delay.push_back(tsp);
    }
    let mut ac_data = Vec::with_capacity((OFDM_FRAME_SYMBOLS + 7) / 8);
    ac_data.resize((OFDM_FRAME_SYMBOLS + 7) / 8, 0xffu8);
    let mut tmcc_data = vec![];
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
    for _ in 0..number_of_carriers * SEGMENTS {
        prbs_table.push((pilot_prbs_state & 1) == 1);
        pilot_prbs_state = pilot_prbs(pilot_prbs_state);
    }
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(symbol_len);
    let mut convolutional_encoder_state = 0;
    (_, convolutional_encoder_state) = convolve(&[TS_SYNC_BYTE], convolutional_encoder_state);
    let mut prbs_state = BYTE_PRBS_INITIAL_STATE;
    let mut segment_carriers = Vec::new();
    loop {
        let mut packet = [0u8; TS_SIZE];
        stdin.read_exact(&mut packet)?;
        let encoded_packet = rs_encoder.encode(&packet);
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
        let convolved;
        (convolved, convolutional_encoder_state) = convolve(&ibuf, convolutional_encoder_state);
        let mut bit_buf = Vec::new();
        for b in puncture(&convolved) {
            bit_delay.push_back(b);
            bit_buf.push(bit_delay.pop_front().unwrap());
        }
        for c in bit_buf.chunks(2) {
            let b0 = c[0];
            let b1 = c[1];
            qpsk_delay.push_back(b1);
            let b1 = qpsk_delay.pop_front().unwrap();
            let symbol = qpsk_mapping[match (b0, b1) {
                (false, false) => 0,
                (true, false) => 1,
                (true, true) => 2,
                (false, true) => 3,
            }];
            segment_carriers.push(symbol);
            if segment_carriers.len() == number_of_data_carriers {
                let mut randomized = Vec::with_capacity(number_of_data_carriers);
                randomized.resize(number_of_data_carriers, Complex32::ZERO);
                for (i, symbol) in segment_carriers.iter().enumerate() {
                    assert_eq!(time_interleaver.index, i);
                    randomized[MODE3_CARRIER_RANDOMIZE[i]] = time_interleaver.push(*symbol)
                }
                let mut carriers = Vec::with_capacity(8192);
                carriers.resize(8192, Complex32::ZERO);
                let off = symbol_len / 2 - number_of_carriers / 2;
                let segment_carries = &mut carriers[off..off + number_of_carriers];
                let seg = 6;
                let tmcc_carriers = TMCC_CARRIER[seg];
                let ac_carriers = AC_CARRIER[seg];
                let mut data_index = 0;
                let segment_carrier_index = number_of_carriers * seg;
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
                        segment_carries[i] = randomized[data_index];
                        data_index += 1;
                    }
                }
                let (negative_freq, positive_freq) = carriers.split_at_mut(symbol_len / 2);
                positive_freq.swap_with_slice(negative_freq);
                ifft.process(&mut carriers);
                let guard_interval = &carriers[carriers.len() - guard_interval_len..];
                {
                    let bytes;
                    unsafe {
                        bytes = std::slice::from_raw_parts(
                            guard_interval.as_ptr() as *const u8,
                            guard_interval.len() * std::mem::size_of::<Complex32>(),
                        );
                    }
                    output.write(bytes)?;
                }
                {
                    let bytes;
                    unsafe {
                        bytes = std::slice::from_raw_parts(
                            carriers.as_ptr() as *const u8,
                            carriers.len() * std::mem::size_of::<Complex32>(),
                        );
                    }
                    output.write(bytes)?;
                }
                ofdm_symbol_index += 1;
                if ofdm_symbol_index == OFDM_FRAME_SYMBOLS - 2 {
                    prbs_state = BYTE_PRBS_INITIAL_STATE;
                }
                if ofdm_symbol_index == OFDM_FRAME_SYMBOLS {
                    ofdm_symbol_index = 0;
                    eprintln!("{ofdm_frame_index}");
                    ofdm_frame_index += 1;
                }
                segment_carriers.clear();
            }
        }
    }
}
