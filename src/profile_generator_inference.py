import numpy as np
import pandas as pd
import os
import shutil
import Bio.SeqIO as SeqIO
import math
import multiprocessing
from npy_append_array import NpyAppendArray
from .paths import PROFILES_DIR, SKIPPED_LOG_NAME, TRINUCLEO, TETRANUCLEO
from colorama import Fore, init
init(autoreset=True)



### This script is essential for the exon prediction pipeline laid out in 'hex_finder.py', as it defines the algorithm that estimates
### structural profiles (based on parameters from Sharma et al., 2025) for the sequences prior to the prediction of exons via the neural network
### A less-optimized (but functionally equivalent) version of this algorithm was used to produce the training set for the models (see the manuscript: https://doi.org/10.64898/2025.12.19.694709)
### The functionality of this script is not currently exposed to the user as a command-line tool, but it could be easily if needed for some reason.
### Multithreading and careful algorithm implementation are used to prevent this step from being a bottleneck compared to exon prediction/inference 
### by the model. More specifically, pre-computed parameters are re-used by the algorithm to navigate and alternate reading frames.



# MODULE PATHS AND CONSTANTS (DO NOT CHANGE UNLESS YOU REALLY KNOW WHAT YOU'RE DOING)
maximum_sequence_length_allowed = 10000000 # TODO: Revisit this number. While technically possible, it may take a while to process and predict for such a long sequence.
profiles_save_path = PROFILES_DIR
log_file_name = SKIPPED_LOG_NAME
window_length = 27
model_input_size = (77, 28) # Locked by model architecture and training
param_table_paths = { 3 : TRINUCLEO, # Path to CSV of 22 TRI-nuncleotide-based physicochemical parameters calculated by Sharma et al., 2025 (see README and manuscript)
                      4 : TETRANUCLEO} # Path to CSV of 6 TETRA-nuncleotide-based physicochemical parameters calculated by Sharma et al., 2025 (see README and manuscript)
script_name = os.path.basename(__file__) # Used globally in print statements to provide the name of this file (useful for debugging when used in pipeline)
skip_seqs_w_Ns = True
placeholder = 'VOID'

# These parameters control multithreading and were chosen after some rudimentary testing 
# They should be safe/reasonably fast on a relatively decent and modern 8+ thread CPU not already under heavy load.
# On anything notably better than that, profile generation should fly by quite quickly with these settings
threads = os.cpu_count() # Used by multiprocessing in profiles_batch()
max_length_before_split = 10000 # Sequences below this length are parallelized by giving workers whole sequences, while those above are processed one at a time and workers are given chunks of a single sequence
threads_to_save = 2 # How many of the CPU threads are left for the system (i.e. unutilized for multithreading)?



### FUNCTION DEFINITIONS

def log_sequence(field1: str,
                 field2: str,
                 log_path: str = PROFILES_DIR / SKIPPED_LOG_NAME):
    """
    Utility function for logging the success/outcome for each sequence during profile generation.
    This provides documentation to the user if sequences were skipped/dropped due to any issues.
    
    Args:
        field1: intended for the sequence id, grabbed from the FASTA header
        field2: status of whether the sequence was processed or not and why
        log_path: Path to write this file tracking skipped sequences (defaults to saving in profiles_save_path)
    """
    
    with open(log_path, mode = 'a') as log:
        log.writelines([f'{field1}\t{field2}\n'])


def load_and_sort_by_length(fasta_path: str, # TODO: Respect lowercase nucleobase masking, perhaps as a user argument for 'hex_finder.py'?
                            min_seq_length: int = int((window_length - 1) + model_input_size[0]), # TODO: Revisit this number. This leaves a single 77 bp window of profile after profile generation, prediction will be made for just one position/nucleotide in the center of the sequence
                            max_seq_length: int = maximum_sequence_length_allowed) -> dict: 
    """
    Uses Biopython functionality to load a FASTA file and group the sequences and ids (headers) by sequence length.
    Also converts seqs to all caps (currently ignores masking) and checks that the sequences contain only valid characters.
    Limits sequences that will be processed by min_seq_length and max_seq_length (min is critical, max is somewhat arbitrary).
    """
    
    # Accepted seq chars
    accept = ['A', 'T', 'C', 'G', 'N']
    accepted = set(accept)
    
    # open the fasta file using biopython
    sequences = list(SeqIO.parse(fasta_path, "fasta"))

    # create empty dictionary for sequence lengths and their corresponding sequences
    lengths = {}

    for record in sequences:
        
        seq = set(str(record.seq).upper()) # convert current sequence into uppercase string, then a set for content checking
        if seq.issubset(accepted): # Does this sequence conform to the accepted characters that the profile generator can handle?
            
            if len(record.seq) >= min_seq_length and len(record.seq) <= max_seq_length:
                length = len(record.seq) # get length of each sequence
                
                if length not in lengths: # check if length already exists in dictionary
                    lengths[length] = [(str(record.seq).upper(), record.id)] # add length as key with value being the id of that particular sequence
                    
                else:
                    lengths[length].append((str(record.seq).upper(), record.id)) # append to existing list under same key (length)
            
            else:
                log_sequence(record.id, f'Skipped: too long (>{max_seq_length} bp) or too short (<{min_seq_length} bp).')
                # print(f'Sequence {record.id} will not be processed due being too long (>{max_seq_length} bp) or too short (<{min_seq_length} bp).')
        
        else:
            log_sequence(record.id, f'Skipped: has characters other than {accept}.')
            # print(f'Sequence {record.id} will not be processed due to unacceptable characters.')
    
    return lengths


def prep_params_table(path_to_csv: str) -> pd.DataFrame:
    """
    Facilitates reading in the parameter table CSVs from Sharma et al. 2025 to a Pandas DataFrame.
    """
    
    params_table = pd.read_csv(path_to_csv, index_col = 0).transpose()
    params_table[placeholder] = np.nan
    return params_table


def split_string(string: str, 
                 split_length: int, 
                 start_frame_idx: int = 0) -> list:
    """
    Splits a sting into consecutive substrings of split_length starting from a chosen location in the string (start_frame_idx)
    This is used by the profile generation algorithm to generate the reading frames required for mapping trimers and tetrameters to numbers.
    """
    
    frame = [string[(start_frame_idx + i):(start_frame_idx + i + split_length)] for i in range(0, len(string), split_length)]
    frame = [ placeholder if len(word) < split_length else word for word in frame ]
    return frame


def prep_slide_indices(step_length: int,
                       seq_length: int) -> tuple:
    """
    Calculates indices used by profile generation loop (in calculate_profile()) that are re-used for a given sequence length, but based
    on the length of the sequence, sliding window length (in bases), and length of the nucleotide step (di-, tri-, tetra-, etc.)
    Calculating these algorithmic parameters in advance is essential to eliminated nested loops and enable fast profile generation in Python.
    """
    
    window_n_nucleosteps = window_length - (step_length - 1) # number of N-nucleotide steps in the chosen window width
    max_frame_idx = math.floor(seq_length / step_length) # greatest index value needed in the reading frame coordination list
    frame_idx = np.arange((max_frame_idx + 1) * step_length) // step_length # list of indices that controls which positions are selected in each reading frame
    which_frame = list(range(step_length)) * math.ceil(seq_length / step_length) # list of indices that controls alternation between the reading frames
    final_seq_length = seq_length - (window_length - 1) # how many window nucleotides/positions do we want to end up with in the final profiles?
    return (frame_idx, which_frame, seq_length, window_n_nucleosteps, final_seq_length)


def calculate_profile(seq: str, 
                      step_length: int, 
                      loop_indices: tuple, 
                      parameters: pd.DataFrame, 
                      debug: bool = False) -> tuple:
    """
    Moves through the sequence one position at a time, calculating the average of each physicochemical parameter for a sliding window of 'window_length' nucleotides 
    (centered around the current base) by splitting the sequence into all possible tri- or tetra-nucleotide steps. For computational efficiency, this is done by 
    splitting the entire sequence into N reading frames of N-nucleotide steps (each offset by one base) and then simultaneously alternating between and incrementing through them. 
    NOTE: For reliability, please only use with window_lengths equal to 9, 15, 21, 27, 33, 39, etc. (i.e. odd multiples of 3). May or may not work outside of those use cases.
    NOTE: This function expects A, T, G, C, or N nucleotide symbols that are all uppercase (remove or convert lowercases/masking first).
    """
    
    if not (step_length in [3, 4]):
        Exception(f'{script_name}: calculate_profiles() only works with step_length equal to 3 or 4!')
    
    # Prepare reading frames and corresponding parameter values (for each position in each frame)
    frames = [ split_string(seq, step_length, start_idx) for start_idx in range(step_length) ]
    
    try: # this try except block handles nucleotide steps with Ns in them by:
        n_nucleo_frames = np.array([np.array(parameters[frame]) for frame in frames])
    except:
        if (skip_seqs_w_Ns): # a) skipping sequences that contain N's or...
            return (None, False)
        else: # b) or averaging all possibilities, i.e. mean of profile for N replaced with A, T, G, and C in the step of interest.
            for frame in frames: # NOTE: This code is not ideal, particularly for more than one N per step, e.g. for cases like NNGC, NNNA, etc.
                for step in frame:
                    if 'N' in step:
                        print(f"{script_name}: N's detected in {step}.") # Make user aware of how the N falls within the steps
                        sum_of_options = np.zeros(parameters.shape[0])
                        for possibility in ['A', 'T', 'G', 'C']:
                            sum_of_options = np.add(sum_of_options, np.array(parameters[step.replace('N', possibility)]))
                        parameters[step] = sum_of_options / 4
            n_nucleo_frames = np.array([np.array(parameters[frame]) for frame in frames])
    
    if debug: # Useful to check that the windows are properly positioned (see loop below)
        frames = np.array(frames)
    
    # Loop through this sequence one base at a time and calculate moving average for sliding window (all parameters)
    (frame_idx, which_frame, seq_length, window_n_nucleosteps, final_seq_length) = loop_indices
    profiles = np.zeros((parameters.shape[0], seq_length - (window_length - 1)))
    for i in range(final_seq_length):
        if debug: # Useful to check that the windows are properly positioned
            print(frames[which_frame[i:(i + window_n_nucleosteps)], frame_idx[i:(i + window_n_nucleosteps)]]) 
        full_window = n_nucleo_frames[ which_frame[i:(i + window_n_nucleosteps)], :, frame_idx[i:(i + window_n_nucleosteps)] ].transpose()
        profiles[:, i] = np.mean(full_window, axis=1)
    
    return (profiles, True)


def calculate_multiframelength_profile(seq_item: tuple) -> tuple:
    """
    Initiates calculation of the profiles using tri- and tetra-nucleotide steps for this sequence and concatenates the results from both into one array.
    """
    
    frame_resources = seq_item[2]
    
    profiles = tuple(calculate_profile(seq=seq_item[0], step_length=frame[0], loop_indices=frame[1], parameters=frame[2]) for frame in frame_resources)
    
    if not any([profile[1] for profile in profiles]): # Was this sequence skipped due to the presence of N's?
        return (seq_item[1], None, False, seq_item[3])
    else:
        merged_profiles = np.concatenate([profile[0] for profile in profiles], axis=0)
        return (seq_item[1], merged_profiles, True, seq_item[3])


def get_frame_resources(seq_length: int, 
                        param_tables: dict) -> list:
    """
    Prepares and packages all resources that will be re-used to process all sequences of a specific sequence length.
    This is integral to a fast profile generation algorithm, which uses pre-baked constants
    and parameters to map k-mers to structural params and calculate a sliding window average efficiently
    """
    
    return [ [step_length, prep_slide_indices(step_length, seq_length), param_tables[step_length]] for step_length in [3,4] ]


def process_long_sequence(seq_tuple: tuple,
                          param_tables: dict,
                          profiles_path: str,
                          chunk_size_target: int,
                          verbose: bool = False,
                          one_file_per_seq: bool = True) -> int:
    """
    Handles the splitting, parallel processing, and stitching of a single long sequence.
    Splits must occur at multiples of 12 (LCM of 3 and 4) to maintain reading frame alignment.
    Merges trailing chunks smaller than window_length into the previous chunk to prevent crashing.
    This function is analogous to profiles_batch() below, except the "batch" is a single sequence split up for multithreading.
    
    Args:
        seq_tuple: A tuple holding the sequence string, sequence id (from the header), and sequence length
        param_tables: A dictionary with integer keys equal to the nucleotide k-mer/step length (3 or 4) and values equal to Pandas data frames with the corresponding mappings from Sharma et al., 2025
        profiles_path: What directory should the structural profiles (NPY files) be saved under?
        chunk_size_target: What is the approximate length to split long sequences up into for parallel processing?
        verbose: Print additional updates/output (useful for debugging)?
        one_file_per_seq: Should one NPY be saved per sequence, or should it be appended to a file of profiles for all sequences of the same length?
    
    Returns:
        An integer indicating exit status (i.e. success: 1/failure: 0).
    """
    seq_seq = seq_tuple[0]
    seq_id = seq_tuple[1]
    seq_len = seq_tuple[2]
    
    # Ensure chunk size is a multiple of 12 (LCM of 3 and 4) to preserve reading frame phase across splits
    chunk_size = (chunk_size_target // 12) * 12
    overlap = window_length - 1 # Amount of context needed from the next chunk to complete the window at the boundary
    typical_chunk_length = chunk_size + overlap - 1
    typical_chunk_resources = get_frame_resources(typical_chunk_length, param_tables)
    
    # Create chunks
    chunks = []
    n_chunks = math.ceil(seq_len / chunk_size)
    for i in range(n_chunks):
        start = i * chunk_size
        # For all chunks except the last, we grab 'overlap' extra bases from the next chunk
        # This ensures the sliding window has data to compute the edge values correctly
        end = min((i + 1) * chunk_size + overlap, seq_len) 
        
        chunk_sub_seq = seq_seq[start:end]
        chunk_real_len = len(chunk_sub_seq)
        
        if chunk_real_len != typical_chunk_length:
            # If this chunk is shorter than the window length (e.g., a tiny remainder),
            # it will cause a crash, so it must be merged with the previous chunk.
            if chunk_real_len < window_length:
                if len(chunks) > 0:
                    # Retrieve the previous chunk
                    prev_chunk = chunks[-1]
                    prev_index = prev_chunk[1]
                    
                    # Calculate where the previous chunk started
                    prev_start = prev_index * chunk_size
                    
                    # Create a new merged sequence from previous start to the absolute end of the current chunk
                    # (Since this usually happens on the last chunk, 'end' is typically seq_len)
                    merged_seq = seq_seq[prev_start:end]
                    merged_len = len(merged_seq)
                    
                    # Generate resources for this new extended chunk
                    merged_resources = get_frame_resources(merged_len, param_tables)
                    
                    # Fix the last chunk (retaining the old previous index)
                    chunks[-1] = (merged_seq, prev_index, merged_resources, merged_len)
                    
                    if verbose:
                        print(f"{script_name}: Merged small tail chunk ({chunk_real_len} bp) into previous chunk for stability.")
                        
                else:
                    if verbose:
                        print(Fore.RED + f'WARNING: {seq_id} appears to be empty or too short to convert to a profile.')
                    log_sequence(seq_id, f"Could not process. Sequence is either empty or too short, but somehow made it past initial length check.")
                    return 0
            else:
                # In this case the chunk is shorter than the typical chunk, but not too short for the algorithm (i.e. < window_length)
                short_chunk_resources = get_frame_resources(chunk_real_len, param_tables)
                chunks.append((chunk_sub_seq, i, short_chunk_resources, chunk_real_len))
        else:
            # Standard processing for valid chunks
            chunks.append((chunk_sub_seq, i, typical_chunk_resources, typical_chunk_length))
            
    if verbose:
        print(f"{script_name}: Parallelizing long sequence {seq_id} ({seq_len} bp) over {len(chunks)} chunks...")
        
    # Process chunks in parallel, i.e. we create a dedicated pool for this sequence
    results = []
    # Using min(threads, len(chunks)) ensures we don't spawn empty workers if chunks < threads
    pool_size = max(1, min(threads, len(chunks)))
    
    with multiprocessing.Pool(processes=pool_size) as pool:
        # map guarantees order, so results[0] is chunk 0, etc.
        results = pool.map(calculate_multiframelength_profile, chunks)
        
    # Check for failures (e.g. Ns)
    if any([not res[2] for res in results]):
        log_sequence(seq_id, f"Skipped: includes N's and algorithm is not equipped to infer parameters.")
        return 0
    
    # Stitch results
    # Each result contains: (chunk_index, profile_array, success, input_len)
    # The profile array has shape (Features, Length). We concatenate along axis 1 (Length).
    try:
        full_profile = np.concatenate([res[1] for res in results], axis=1)
        
        # Verify length consistency
        # Expected output length = Original Length - (Window - 1)
        expected_len = seq_len - overlap
        if full_profile.shape[1] != expected_len:
             print(Fore.RED + f"{script_name}: Warning! Stitched profile length {full_profile.shape[1]} does not match expected {expected_len} for {seq_id}.")
             
        if one_file_per_seq:
            np.save(profiles_path / f'{seq_id}.npy', full_profile)
        else:
            # If not one file per seq, we append to the bulk file for this total length
            with NpyAppendArray(profiles_path / f'{seq_len}bp_seq_profiles.npy', delete_if_exists=False) as npy_file:
                npy_file.append(np.reshape(full_profile, shape=(1, *full_profile.shape)))
            
            with open(profiles_path / f'{seq_len}bp_seq_ids.txt', mode='a') as file:
                file.writelines(f'{seq_id}\n')
                
        return 1
    
    except Exception as e:
        print(Fore.RED + f"{script_name}: Error stitching chunks for {seq_id}: {e}")
        return 0


def profiles_batch(seq_items: list,
                   frame_resources: dict,
                   profiles_path: str,
                   report_iter: int = 10, 
                   save_threads: int = 0, 
                   verbose: bool = False,
                   one_file_per_seq: bool = False):
    """
    This function handles the delegation of profile generation for a batch of sequences via multithreading.
    The resulting Numpy arrays are saved to disk as one or more .NPY files, depending on the operating mode.
    The Python library multiprocessing is used to divide the work over multiple CPU threads for faster run times
    The following resources were helpful in figuring out how to use multiprocessing.Pool() and imap() properly:
    https://stackoverflow.com/questions/14723458/distribute-many-independent-expensive-operations-over-multiple-cores-in-python
    https://superfastpython.com/multiprocessing-pool-imap/
    https://superfastpython.com/multiprocessing-pool-python/
    https://stackoverflow.com/questions/53306927/chunksize-irrelevant-for-multiprocessing-pool-map-in-python
    
    Args:
        seq_items: A list of tuples of the form (sequence, id, length).
        frame_resources: A list of data structures and precomputed algorithm parameters prepared by get_frame_resources() 
            that are required for profile calculation, but stay the same for a given sequence length. Computing these
            parameters once per sequence length, rather than on-the-fly, saves an immense amount of time!
        profiles_path: Path to the directory where the resulting profiles/NPY files should be saved.
        report_iter: After how many completed profiles should an update be printed (useful for debugging).
        save_threads: How many cores are left untouched by the pool (e.g. for system stability)?
        verbose: Should updates be printed to the console at all?
        one_file_per_seq: Are profiles appended to the same NPY file or is a file created per-profile/sequence?
    """
    
    seq_items = list(set(seq_items)) # Eliminate duplicates
    seq_items = [ (seq_tuple[0], seq_tuple[1], frame_resources[seq_tuple[2]], seq_tuple[2]) for seq_tuple in seq_items ]
    N_completed = 0
    
    # Determine the parallelization strategy based on the size of this batch
    if len(seq_items) >= (threads - save_threads):
        processes = (threads - save_threads)
        if len(seq_items) >= (threads - save_threads)^2:
            chunksize = (threads - save_threads)
        else:
            chunksize = len(seq_items) // processes
    else:
        processes = len(seq_items)
        chunksize = 1
    
    # Process the sequences in parallel
    with multiprocessing.Pool(processes = processes) as pool:
        
        for result in pool.imap(func = calculate_multiframelength_profile, iterable = seq_items, chunksize = chunksize):
            if (result[2]):
                
                if one_file_per_seq:
                    
                    # Save the physicochemical profile for this sequence to a file named after the header from the FASTA file
                    np.save(profiles_path / f'{result[0]}.npy', result[1])
                
                else:
                    
                    # Append this profile to the NPY file with others of the same length
                    with NpyAppendArray(profiles_path / f'{result[3]}bp_seq_profiles.npy', delete_if_exists=False) as npy_file:
                        npy_file.append(np.reshape(result[1], shape=(1, *result[1].shape)))
                
                    # Note the order of the sequences saved in the NPY file (ids are the headers from the FASTA)
                    with open(profiles_path / f'{result[3]}bp_seq_ids.txt', mode='a') as file:
                        file.writelines(f'{result[0]}\n')
                
                N_completed += 1
                
                if verbose and ((N_completed % report_iter) == 0):
                    print(f"{script_name}: {N_completed} profiles completed, {round(N_completed/len(seq_items)*100, 1)}% of this batch.")
            else:
                log_sequence(result[0], f"Skipped: includes N's and algorithm is not equipped to infer parameters.")
                if verbose:
                    print(Fore.YELLOW + f"{script_name}: sequence {result[0]} skipped due to N's.")


def generate_profiles(input_FASTA_path: str,
                      param_table_paths: dict = param_table_paths,
                      profiles_path: str = profiles_save_path,
                      clear_old_profiles: bool = True,
                      threads_to_save = threads_to_save,
                      verbose: bool = False,
                      one_file_per_seq: bool = True,
                      parallel_threshold: int = max_length_before_split) -> list:
        """
        This function initiates structural profile calculation for the sequences provided in a FASTA file and saves the profiles.
        It sorts sequences into "long" ones, that will need to be parallelized by splitting each sequence up over workers, and
        "short" ones that can be handed out to workers in their entirety. This ensures better speed/pool utilization efficiency.
        While this function may be repurposed, current defaults are required for calling in 'hex_finder.py' to work properly.
        
        Args:
            input_FASTA_path: Path to a properly formatted FASTA file with headers (staring with '>') followed by either 
                single line sequences or consistently wrapped, multi-line sequences. The entire header is taken as the 
                ID for a sequence, which the NPY/profile files are named after, so format your headers to be concise and unique.
            param_table_paths: Path to the CSV files that map tri-nucleotides and tetra-nucleotides to physicochemical parameter values.
            profiles_path: Path to a directory in which the structural profiles (NPY files) will be saved.
            clear_old_profiles: Boolean indicating whether to delete any files currently in the profiles_path directory.
            threads_to_save: The profile generation algorithm uses multithreading to speed things up, how many threads should be reserved/unused?
            verbose: Flag controlling how much output should be printed to the console during profile generation.
            one_file_per_seq: Should the structural profile for each sequence be saved in separate NPY files or appended to the same one?
                While 'False' technically works, 'hex_finder.py' and 'visualize_predictions.py' currently expect output generated using 'True'.
            parallel_threshold: Sequences longer than this integer are processed "individually" by splitting them up for the pool. Sequences shorter
                than this integer are given to a worker whole. This ensures more efficient threading/worker utilization.
        
        Returns:
            A list of tuples, one tuple per sequence, with (sequence, id, length). This is useful for external inference ETC estimation later on, among other things.
        """
        
        # Delete all existing structural profiles before proceeding
        if clear_old_profiles and os.path.exists(profiles_path):
            shutil.rmtree(profiles_path)
            os.mkdir(profiles_path)
        
        # Pre-load parameter tables once to pass to sub-functions
        loaded_param_tables = { step: prep_params_table(path) for step, path in param_table_paths.items() }
        
        # Determine the sequence lengths contained in the FASTA file
        seq_lengths = load_and_sort_by_length(input_FASTA_path)
        
        # This is the final list that will be returned, containing metadata for ALL processed sequences
        seqs_unpacked = []
        
        if len(seq_lengths) != 0:
            
            # Sub-list specifically for the standard batch processor
            standard_seqs_unpacked = []
            frame_resources = {} 
            
            for seq_length in seq_lengths:
                
                # Check if this length group qualifies as "Long"
                if seq_length >= parallel_threshold:
                    # Process these individually
                    for seq_tuple in seq_lengths[seq_length]:
                        # Create the tuple: (sequence, id, length)
                        long_seq_item = (seq_tuple[0], seq_tuple[1], seq_length)
                        
                        # Process immediately
                        success = process_long_sequence(long_seq_item,
                                                        param_tables=loaded_param_tables,
                                                        profiles_path=profiles_path,
                                                        verbose=verbose,
                                                        one_file_per_seq=one_file_per_seq,
                                                        chunk_size_target=parallel_threshold)
                        
                        # If successful, add to the final return list
                        if success:
                            seqs_unpacked.append(long_seq_item)
                            
                else:
                    # Add to standard batch lists
                    # Initialize resources for this length if not done yet
                    if seq_length not in frame_resources:
                        frame_resources[seq_length] = get_frame_resources(seq_length, loaded_param_tables)
                    
                    for seq_tuple in seq_lengths[seq_length]:
                        item = (seq_tuple[0], seq_tuple[1], seq_length)
                        
                        # Add to the sub-list for batch processing
                        standard_seqs_unpacked.append(item)
                        # Add to the final return list
                        seqs_unpacked.append(item)
            
            # Process the standard sequences using the original batch method
            if len(standard_seqs_unpacked) > 0:
                profiles_batch(standard_seqs_unpacked, 
                               frame_resources = frame_resources,
                               verbose=verbose,
                               profiles_path=profiles_path,
                               save_threads=threads_to_save,
                               one_file_per_seq=one_file_per_seq)
            
            # Return the unified list of all sequences (both long and standard)
            return seqs_unpacked 
        
        else:
            print(Fore.RED + f'No valid sequences were found in {input_FASTA_path}, please check file for proper formatting and base characters.')
            return []