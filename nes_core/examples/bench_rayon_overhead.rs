//! Measures rayon overhead: empty par_iter vs serial iter at 16 workers.
//! If this reports meaningful overhead, rayon's scheduler is the tax;
//! if not, the overhead in step_all is alloc/collect/tuple packing.

use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

fn main() {
    let n_workers = 16;
    let n_steps = 10_000;

    let workers: Vec<Mutex<u64>> = (0..n_workers).map(|i| Mutex::new(i as u64)).collect();

    // Warmup.
    for _ in 0..1000 {
        workers.par_iter().for_each(|mu| {
            let mut g = mu.lock().unwrap();
            *g = g.wrapping_add(1);
        });
    }

    // 1) Pure-rayon empty closure — measures join+dispatch overhead only.
    let t0 = Instant::now();
    for _ in 0..n_steps {
        workers.par_iter().for_each(|_| {});
    }
    let t_empty = t0.elapsed();

    // 2) Tiny closure (mutex lock + single u64 bump).
    let t0 = Instant::now();
    for _ in 0..n_steps {
        workers.par_iter().for_each(|mu| {
            let mut g = mu.lock().unwrap();
            *g = g.wrapping_add(1);
        });
    }
    let t_lock = t0.elapsed();

    // 3) par_iter + map + collect into Vec<u64>.
    let t0 = Instant::now();
    for _ in 0..n_steps {
        let _v: Vec<u64> = workers
            .par_iter()
            .map(|mu| {
                let mut g = mu.lock().unwrap();
                *g = g.wrapping_add(1);
                *g
            })
            .collect();
    }
    let t_collect = t0.elapsed();

    // 4) par_iter + map returning a big tuple (simulates Pool::step_all).
    const FRAME_PIXELS: usize = 240 * 256;
    let t0 = Instant::now();
    for _ in 0..n_steps {
        let _v: Vec<(Vec<u8>, Vec<u8>, Vec<u8>, bool)> = workers
            .par_iter()
            .map(|mu| {
                let _g = mu.lock().unwrap();
                (
                    vec![0u8; FRAME_PIXELS * 3],
                    vec![0u8; 84 * 84],
                    vec![0u8; 2048],
                    false,
                )
            })
            .collect();
    }
    let t_tuple = t0.elapsed();

    println!("rayon overhead bench ({} workers × {} iters):", n_workers, n_steps);
    println!("  empty par_iter               {:6.2} µs/iter", t_empty.as_micros() as f64 / n_steps as f64);
    println!("  par_iter + mutex+bump        {:6.2} µs/iter", t_lock.as_micros() as f64 / n_steps as f64);
    println!("  par_iter + map + collect     {:6.2} µs/iter", t_collect.as_micros() as f64 / n_steps as f64);
    println!("  par_iter + tuple-alloc       {:6.2} µs/iter", t_tuple.as_micros() as f64 / n_steps as f64);
}
