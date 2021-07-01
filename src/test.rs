use once_cell::sync::{Lazy}; // 1.7.2
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time;

static GLOBAL_LOCK: Lazy<Mutex<AtomicIsize>> = Lazy::new(|| Mutex::new(AtomicIsize::new(0)));

fn main() {

let thread_1 = thread::spawn(move || {
    // some work here
    let mut i = 0;

    while i < 10 {
        let has_lock = GLOBAL_LOCK.lock().unwrap();
        thread::sleep(time::Duration::from_millis(1));
        println!("Thread 1: iteration {}", i);
        i = i + 1;
    }
});
let thread_2 = thread::spawn(move || {
    // some work here
    let mut i = 0;
    
    while i < 10 {
    let has_lock = GLOBAL_LOCK.lock().unwrap();
    thread::sleep(time::Duration::from_millis(2));
        println!("Thread 2: iteration {}", i);
        i = i + 1;
    }
});

// some work here
let res1 = thread_1.join();
let res2 = thread_2.join();
    
}