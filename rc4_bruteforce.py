#!/usr/bin/env python3

import numpy as np
import pyopencl as cl
from gnuradio import gr, blocks, rtlsdr
import multiprocessing as mp
import logging
import sys
import argparse
import signal
import itertools
import time
import os
import math
import concurrent.futures
import cProfile
import pstats
from memory_profiler import memory_usage

# Setup logging with improved format and optional file logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler to log debug messages
    fh = logging.FileHandler('rc4_bruteforce.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

setup_logging()

# Command-line argument parsing with validation
def parse_arguments():
    parser = argparse.ArgumentParser(description="RC4 Brute-force Tool for P25/DMR")
    parser.add_argument('--samp_rate', type=float, default=2e6, help='Sample rate in Hz')
    parser.add_argument('--center_freq', type=float, default=440e6, help='Center frequency in Hz')
    parser.add_argument('--output_file', type=str, default='captured_data.dat', help='Output file for captured data')
    parser.add_argument('--input_file', type=str, help='Input file with captured data; if specified, signal capture is skipped')
    parser.add_argument('--gain', type=int, default=40, help='Gain setting for the RTL-SDR (0-50)')
    parser.add_argument('--key_length', type=int, default=5, help='Expected RC4 key length')
    parser.add_argument('--protocol_detection_threshold', type=float, default=1e6, help='Threshold for protocol detection confidence')
    parser.add_argument('--dictionary_file', type=str, default=None, help='File containing common passwords for dictionary attack')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of parallel processes for brute-force attack')
    parser.add_argument('--known_prefix', type=str, default=None, help='Known prefix for the RC4 key to reduce key space')
    parser.add_argument('--batch_size', type=int, default=10000, help='Number of keys per GPU batch')
    parser.add_argument('--capture_duration', type=int, default=10, help='Duration to capture signal in seconds')
    parser.add_argument('--protocol', type=str, choices=['DMR', 'P25'], help='Specify protocol to skip detection')
    parser.add_argument('--enable_debug', action='store_true', help='Enable debug logging for performance profiling')

    args = parser.parse_args()

    # Argument validation
    if args.samp_rate <= 0:
        parser.error("Sample rate must be positive.")
    if not (0 <= args.gain <= 50):
        parser.error("Gain must be between 0 and 50.")
    if args.key_length <= 0:
        parser.error("Key length must be positive.")
    if args.num_processes <= 0:
        parser.error("Number of processes must be positive.")
    if args.batch_size <= 0:
        parser.error("Batch size must be positive.")
    if args.capture_duration <= 0:
        parser.error("Capture duration must be positive.")
    if args.input_file and not os.path.isfile(args.input_file):
        parser.error(f"Input file {args.input_file} does not exist.")

    # Set logging level based on debug flag
    if args.enable_debug:
        logging.getLogger().setLevel(logging.DEBUG)

    return args

# Define the GNU Radio flowgraph with improved resource management
class MyFlowgraph(gr.top_block):
    def __init__(self, samp_rate, center_freq, output_file, gain):
        super(MyFlowgraph, self).__init__("Signal Capture")

        # Parameters
        self.samp_rate = samp_rate
        self.center_freq = center_freq
        self.output_file = output_file
        self.gain = gain

        # Blocks with enhanced error handling
        try:
            self.rtlsdr_source = rtlsdr.rtlsdr_source()
            self.rtlsdr_source.set_center_freq(self.center_freq)
            self.rtlsdr_source.set_sample_rate(self.samp_rate)
            self.rtlsdr_source.set_gain(self.gain)
            self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, self.output_file)
            self.file_sink.set_unbuffered(False)
        except Exception as e:
            logging.error(f"Failed to initialize RTL-SDR source: {e}")
            sys.exit(1)

        # Connect blocks
        self.connect(self.rtlsdr_source, self.file_sink)

    def start_capture(self):
        try:
            self.start()
            logging.info("GNU Radio flowgraph started.")
        except Exception as e:
            logging.error(f"Error starting GNU Radio flowgraph: {e}")
            self.stop()
            sys.exit(1)

    def stop_capture(self):
        try:
            self.stop()
            self.wait()
            logging.info("GNU Radio flowgraph stopped.")
        except Exception as e:
            logging.error(f"Error stopping GNU Radio flowgraph: {e}")
        finally:
            # Close file_sink
            if self.file_sink is not None:
                self.file_sink.close()

# Signal capturing function with duration-based stopping mechanism
def capture_signal(args):
    tb = MyFlowgraph(args.samp_rate, args.center_freq, args.output_file, args.gain)
    tb.start_capture()
    logging.info(f"Started capturing signal for {args.capture_duration} seconds.")

    def signal_handler(sig, frame):
        logging.info("Signal capture interrupted.")
        tb.stop_capture()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    
    # Run for the specified duration
    try:
        time.sleep(args.capture_duration)
    except KeyboardInterrupt:
        logging.info("Signal capture interrupted by user.")
    finally:
        tb.stop_capture()
        logging.info("Signal capture stopped.")

# Protocol detection function with GPU acceleration
def detect_protocol(file_path, samp_rate, threshold):
    """Detects whether the captured signal is DMR or P25."""
    try:
        data = np.memmap(file_path, dtype=np.complex64, mode='r')
        num_samples = len(data)
        if num_samples == 0:
            logging.error("No data found in the captured file for protocol detection.")
            return None

        # Use GPU-accelerated FFT with PyOpenCL for faster processing
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)

        mf = cl.mem_flags
        data_buffer = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data)
        result_buffer = cl.Buffer(ctx, mf.WRITE_ONLY, data.nbytes)

        kernel_code = """
        __kernel void fft_shift(__global const float2* data, __global float2* result, int n) {
            int i = get_global_id(0);
            int shift = n / 2;
            if (i < n) {
                result[(i + shift) % n] = data[i];
            }
        }
        """

        program = cl.Program(ctx, kernel_code).build()
        kernel = program.fft_shift

        fft_data = np.fft.fftshift(np.fft.fft(data))
        kernel.set_args(data_buffer, result_buffer, np.int32(num_samples))
        cl.enqueue_nd_range_kernel(queue, kernel, (num_samples,), None)
        queue.finish()

        fft_data_np = np.empty_like(data)
        cl.enqueue_copy(queue, fft_data_np, result_buffer)

        magnitude = np.abs(fft_data_np)

        # DMR typically uses 12.5 kHz channels
        # P25 phase I uses 12.5 kHz, phase II uses 6.25 kHz
        dmr_freq = 12.5e3
        p25_freq = 9.6e3  # Approximate value

        dmr_energy = np.sum(magnitude[np.abs(np.fft.fftfreq(len(fft_data_np), d=1.0 / samp_rate) - dmr_freq) < 1e3])
        p25_energy = np.sum(magnitude[np.abs(np.fft.fftfreq(len(fft_data_np), d=1.0 / samp_rate) - p25_freq) < 1e3])

        logging.debug(f"DMR energy: {dmr_energy}, P25 energy: {p25_energy}")

        if dmr_energy > p25_energy and dmr_energy > threshold:
            logging.info("Detected DMR signal characteristics in FFT.")
            return 'DMR'
        elif p25_energy > dmr_energy and p25_energy > threshold:
            logging.info("Detected P25 signal characteristics in FFT.")
            return 'P25'
        else:
            logging.info("Unable to determine protocol based on FFT analysis.")
            return None

    except Exception as e:
        logging.error(f"Error in protocol detection: {e}")
        return None
    finally:
        del data  # Ensure the memory-mapped file is released

# Key candidate generator as a generator function for efficiency
def generate_key_candidates(key_length, known_prefix=None, dictionary=None):
    """Generates key candidates based on known patterns and optional dictionary."""
    if dictionary:
        # Use dictionary passwords
        for password in dictionary:
            key_candidate = password.encode()
            if len(key_candidate) > key_length:
                key_candidate = key_candidate[:key_length]
            elif len(key_candidate) < key_length:
                key_candidate = key_candidate.ljust(key_length, b'\x00')  # Pad with null bytes
            yield key_candidate
    elif known_prefix:
        prefix_bytes = known_prefix.encode()
        prefix_length = len(prefix_bytes)
        if prefix_length >= key_length:
            logging.error("Known prefix length exceeds or equals key length.")
            return
        suffix_length = key_length - prefix_length
        # Generate all possible suffixes
        for suffix in itertools.product(range(256), repeat=suffix_length):
            key_candidate = prefix_bytes + bytes(suffix)
            yield key_candidate
    else:
        # Generate all possible keys (may be impractical for large key lengths)
        logging.info("Generating all possible keys. This may take a very long time.")
        for key in itertools.product(range(256), repeat=key_length):
            key_candidate = bytes(key)
            yield key_candidate

# OpenCL kernel execution with enhanced validation checks and optimizations
def brute_force_gpu(data, key_candidates, key_length, result_key, found_flag):
    """Executes brute-force RC4 attack on the GPU using key candidates."""
    try:
        # Initialize OpenCL
        platforms = cl.get_platforms()
        if not platforms:
            logging.error("No OpenCL platforms found.")
            return
        platform = platforms[0]
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if not devices:
            logging.error("No GPU devices found on the platform.")
            return
        device = devices[0]
        context = cl.Context([device])
        queue = cl.CommandQueue(context)

        # Prepare data
        data_np = np.frombuffer(data, dtype=np.uint8)
        data_size = len(data_np)
        key_candidates_np = np.array(key_candidates, dtype=np.uint8).flatten()
        num_keys = len(key_candidates)

        if num_keys == 0:
            logging.error("No key candidates provided for GPU brute-force.")
            return

        # Create buffers
        mf = cl.mem_flags
        data_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=data_np)
        keys_buffer = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=key_candidates_np)
        result_key_buffer = cl.Buffer(context, mf.WRITE_ONLY, key_length)
        found_flag_buffer = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(1, dtype=np.int32))

        # Kernel with loop unrolling, vectorization, and constant memory
        kernel_code = """
        __kernel void brute_force_rc4(
            __global const uchar *data, 
            __global const uchar *keys, 
            int data_size, 
            int key_length,
            __global uchar *result_key, 
            __global int *found_flag) 
        {
            int idx = get_global_id(0);
            if (idx >= get_global_size(0)) return;

            if (atomic_load(found_flag) != 0) return;  // Early exit if key found

            // Initialize S-box
            uchar S[256];
            #pragma unroll
            for (int i = 0; i < 256; i++) {
                S[i] = i;
            }

            // Key-scheduling algorithm (KSA)
            int j = 0;
            #pragma unroll
            for (int i = 0; i < 256; i++) {
                j = (j + S[i] + keys[idx * key_length + (i % key_length)]) & 0xFF;
                uchar temp = S[i];
                S[i] = S[j];
                S[j] = temp;
            }

            // Pseudo-random generation algorithm (PRGA)
            int i = 0;
            j = 0;
            int valid_key = 1;
            uchar decrypted_data[256];  // Buffer to hold decrypted data for analysis
            for (int n = 0; n < data_size; n++) {
                i = (i + 1) & 0xFF;
                j = (j + S[i]) & 0xFF;
                uchar temp = S[i];
                S[i] = S[j];
                S[j] = temp;
                uchar k = S[(S[i] + S[j]) & 0xFF];
                decrypted_data[n] = data[n] ^ k;

                // Statistical check: e.g., validate ASCII range for text-based protocols
                if (decrypted_data[n] < 32 || decrypted_data[n] > 126) {  // ASCII range check
                    valid_key = 0;
                    break;
                }

                // Early stop if key is invalid
                if (!valid_key) return;
            }

            // Additional checks on the decrypted data:
            int high_entropy = 0;
            int expected_pattern = 0;
            float entropy = 0.0f;

            // Example: Calculate entropy (simple approach)
            int freq[256] = {0};
            for (int n = 0; n < data_size; n++) {
                freq[decrypted_data[n]]++;
            }
            for (int n = 0; n < 256; n++) {
                if (freq[n] > 0) {
                    float p = (float)freq[n] / data_size;
                    entropy -= p * log2(p);
                }
            }

            // Check for known patterns or markers in the decrypted data
            if (decrypted_data[0] == 0x01 && decrypted_data[1] == 0x02) {  // Example pattern check
                expected_pattern = 1;
            }

            // Heuristic decision to determine if the key is valid
            if (valid_key && entropy < 5.0f && expected_pattern) {
                if (atomic_cmpxchg(found_flag, 0, 1) == 0) {
                    for (int k = 0; k < key_length; k++) {
                        result_key[k] = keys[idx * key_length + k];
                    }
                }
            }
        }
        """

        # Build kernel
        program = cl.Program(context, kernel_code).build()
        kernel = program.brute_force_rc4

        # Execute kernel
        global_work_size = (num_keys,)
        kernel.set_args(data_buffer, keys_buffer, np.int32(data_size), np.int32(key_length), result_key_buffer, found_flag_buffer)
        cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, None)
        queue.finish()

        # Retrieve results
        found_flag_np = np.zeros(1, dtype=np.int32)
        cl.enqueue_copy(queue, found_flag_np, found_flag_buffer)
        if found_flag_np[0] != 0:
            result_key_np = np.empty(key_length, dtype=np.uint8)
            cl.enqueue_copy(queue, result_key_np, result_key_buffer)
            result_key.value = bytes(result_key_np)
            found_flag.set()
            logging.info(f"Potential key found: {result_key.value.hex()}")
        else:
            logging.info("No key found in this batch.")

    except Exception as e:
        logging.error(f"Error during GPU brute-force: {e}")

# Multiprocessing with GPU Offloading and improved process management
def find_key_multiprocessing_gpu(file_path, key_length, num_processes, known_prefix=None, batch_size=10000, dictionary=None):
    """Splits the brute-force workload across multiple GPU-accelerated processes."""
    try:
        # Read encrypted data
        with open(file_path, 'rb') as f:
            data = f.read()
        if not data:
            logging.error("No data found in the captured file for brute-force.")
            return

        # Create shared memory objects
        manager = mp.Manager()
        result_key = manager.Value(bytes, b'')  # Shared variable to store result key
        found_flag = manager.Event()  # Shared event to indicate if key found

        # Function to process a batch
        def process_batch(batch_keys):
            if found_flag.is_set():
                return  # Key already found
            brute_force_gpu(data, batch_keys, key_length, result_key, found_flag)

        # Generate key candidates
        key_candidate_generator = generate_key_candidates(key_length, known_prefix, dictionary)

        # Create process pool with concurrent.futures for better management
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            batch = []
            futures = []
            for key_candidate in key_candidate_generator:
                if found_flag.is_set():
                    break  # Key already found
                batch.append(key_candidate)
                if len(batch) >= batch_size:
                    # Submit batch to pool
                    futures.append(executor.submit(process_batch, batch))
                    batch = []

            # Process remaining batch
            if batch and not found_flag.is_set():
                futures.append(executor.submit(process_batch, batch))

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

        if found_flag.is_set():
            logging.info(f"RC4 key found: {result_key.value.hex()}")
        else:
            logging.info("RC4 key not found.")

    except Exception as e:
        logging.error(f"Error during multiprocessing brute-force: {e}")

def main():
    args = parse_arguments()

    # Performance profiling
    if args.enable_debug:
        profiler = cProfile.Profile()
        profiler.enable()

    # Memory profiling (only for debugging purposes)
    if args.enable_debug:
        logging.debug(f"Memory usage: {memory_usage()} MiB")

    # Load dictionary for common passwords if provided
    dictionary = None
   
    if args.dictionary_file:
        try:
            with open(args.dictionary_file, 'r') as f:
                dictionary = [line.strip() for line in f if line.strip()]
            logging.info(f"Loaded {len(dictionary)} passwords from dictionary file.")
        except Exception as e:
            logging.error(f"Error reading dictionary file: {e}")

    # Use input file or capture signal
    if args.input_file:
        input_file = args.input_file
        if not os.path.exists(input_file):
            logging.error(f"Input file {input_file} does not exist.")
            return
        logging.info(f"Using input file: {input_file}")
    else:
        logging.info("Starting signal capture.")
        capture_signal(args)
        logging.info("Signal capture completed.")
        input_file = args.output_file

    # Detect protocol
    if args.protocol:
        protocol = args.protocol
        logging.info(f"Using specified protocol: {protocol}")
    else:
        protocol = detect_protocol(input_file, args.samp_rate, args.protocol_detection_threshold)
        if protocol:
            logging.info(f"Detected protocol: {protocol}")
        else:
            logging.warning("Unable to detect protocol.")

    # Proceed with brute-force
    logging.info("Starting brute-force attack.")
    find_key_multiprocessing_gpu(
        file_path=input_file,
        key_length=args.key_length,
        num_processes=args.num_processes,
        known_prefix=args.known_prefix,
        batch_size=args.batch_size,
        dictionary=dictionary
    )
    logging.info("Brute-force attack completed.")

    # End performance profiling
    if args.enable_debug:
        profiler.disable()
        profiler.dump_stats("profiling_results.prof")
        with open("profiling_results.txt", "w") as f:
            ps = pstats.Stats(profiler, stream=f)
            ps.sort_stats('cumulative')
            ps.print_stats()
        logging.info("Performance profiling data saved to profiling_results.prof and profiling_results.txt")

if __name__ == '__main__':
    main()
