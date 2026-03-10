### This is main script, which provides command-line access to the exon-prediction models and post-processing pipeline that were created/trained/evaluated
### in the work described by the pre-print (https://doi.org/10.64898/2025.12.19.694709). There are two additional tools that facilitate fetching sequences
### and reference annotation and later visualizing HEX-finder predictions right next to RefSeq features for the same sequence (if present). These tools have been
### provided so that the performance of this method can be transparently and conveniently assessed ('get_demo_seqs.py" and "visualize_predictions.py, see REAME).
### That said, all that is required to use this script is a FASTA containing the target sequences, with unique sequence IDs at the start of each header (between '>' and whitespace). 
### The output will be a series of GFF files (one per sequence) that record the exons predicted for each (if any). These will be saved in '../predictions/'.
### NOTE: The '../predictions/' and '../profiles/' directories are cleared prior to each new run, so move the contents of those folder elsewhere between runs
### if you are running predictions for more than one batch of sequences and want to save the contents of either of these directories for later.



#TODO: Improve efficiency of Python post-processing algorithm (e.g. exon prediction filtering steps), this is the last easily addressable speed bottleneck.
#TODO: Tensorflow output silencing improvements.
#TODO: Currently 'profile_generator_inference.py' is capable of grouping profiles into NPY files by sequence length, support this on the front end?
#TODO: Add a structural profile visualization script?



# This block keeps expectations aligned between HEX-finder and visualize_predictions.py for sequences with no predictions.
# No matter what changes are made to NO_PREDICTIONS_TEMPLATE, there must always be a 'sequence_id' and 'sequence_length' field
# However, where those appear in the string or what other fields are specified in the template does not matter

NO_PREDICTIONS_TEMPLATE = "HEX-finder made no exon predictions for sequence {sequence_id} of length {sequence_length} bp."

def parse_template(template: str) -> tuple:
    """
    Finds all variable names in the template string, creates a regex pattern that captures the variables,
    and then creates a dictionary mapping variable names to group numbers (returns pattern and dict)
    """
    
    import re

    variable_names = re.findall(r'\{(\w+)\}', template)
    pattern = re.escape(template)
    for variable_name in variable_names:
        pattern = pattern.replace( '\{' + variable_name + '\}' , '(.+)')
    group_dict = {name: i+1 for i, name in enumerate(variable_names)}
    
    return re.compile(pattern), group_dict

NO_PREDICTIONS_PATTERN, NO_PREDICTIONS_GROUPS = parse_template(NO_PREDICTIONS_TEMPLATE)



def main (args):
    
    from .paths import MODELS_DIR, PROFILES_DIR, PREDICTIONS_DIR, NORM_DATA_PATHS, EXON_LENGTHS_DIST, NO_PREDICTIONS_LOG, shorten_path
    import os
    import shutil

    # Tensorflow warning suppression (see tf_quiet module), still doesn't work fully in every environment. 
    # handles C-level logs (fd redirection) and the OneDNN flag correctly.
    # This needs to be the first time TensorFlow is touched in the entire process.
    from .tf_quiet import import_tf_quietly
    tf = import_tf_quietly()

    # Module imports can proceed normally now (post silencing)
    from keras.utils import timeseries_dataset_from_array
    from . import keras_models
    from . import profile_generator_inference as pgi
    import numpy as np
    import pandas as pd
    import time
    import json
    from colorama import Fore, init
    init(autoreset=True)
    
    
    
    # Argument assignment to script-wide variables
    exon_level_threshold = args.threshold
    model_name = args.model
    input_fasta = args.fasta
    delete_profiles_after = args.d
    
    
    
    ### SCRIPT-WIDE PATHS, DATA, CONSTANTS (DO NOT CHANGE)
    
    # Project paths
    models_dir = MODELS_DIR
    profiles_dir = PROFILES_DIR
    predictions_dir = PREDICTIONS_DIR
    norm_data_path = NORM_DATA_PATHS
    no_predictions_log = NO_PREDICTIONS_LOG
    
    # Import exon length distribution data for exon-level prediction re-ranking
    length_dist = pd.read_csv(EXON_LENGTHS_DIST)
    length_dist.Proportion = length_dist.Proportion / 100
    
    # Constants
    model_input_size = (77, 28) # DO NOT CHANGE (set by model training)
    final_predictions_offset = int((pgi.window_length-1)/2 + (model_input_size[0]-1)/2 + 1) 
    n_model_classes = 3 # DO NOT CHANGE (set by model training)
    bin_size = 10000 # Size of bins for grouping sequences by length (in bp)
    
    
    
    ### FUNCTION DEFINITIONS
    
    def load_json(path: str,
                verbose: bool = False) -> dict:
        """
        Reads in a JSON file
        """
        
        try:
            with open(path, mode='r') as file:
                jsn = json.load(file)
                if verbose:
                    print(f"Loaded {path}.")
                return jsn
            
        except FileNotFoundError:
            print(Fore.RED + f"ERROR: Input file not found at '{path}'.")
            
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}")


    def prepare_model(model_name: str, 
                    training_input_shape: tuple = model_input_size,
                    n_training_classes: int = n_model_classes,
                    verbose: bool = True):
        """
        Loads one of the three trained models given its name. Returns the loaded Keras model with weights.
        """
        
        weights_path = models_dir / f'{model_name}_best.keras'
        args_path = models_dir / f'{model_name}_training_call.json'
        model_init = load_json(args_path, verbose=False)["model_args"]
        model_init['print_summary'] = False
        
        match model_name:
            case 'TCN':
                model = keras_models.TCN_classifier(input_shape=training_input_shape, n_classes=n_training_classes, **model_init)
            case 'BiLSTM':
                model = keras_models.LSTM_classifier(input_shape=training_input_shape, n_classes=n_training_classes, **model_init)
            case 'MBDA-Net':
                model = keras_models.MBDA_Net(input_shape=training_input_shape, n_classes=n_training_classes, **model_init)
            case _:
                print(Fore.RED + f"ERROR: {model_name} must be either 'TCN', 'BiLSTM', or 'MBDA-Net'.")
                exit()
        model.load_weights(weights_path)
        
        if verbose:
            print(Fore.GREEN + f'Loaded model ({model_name}) weights and architecture.')
        
        return model


    def extrude_1D(vector: np.ndarray, 
                n_columns: int) -> np.ndarray:
        """
        Takes a 1D Numpy array and copies it repeatedly along a new dimension.
        """
        
        if len(vector.shape) != 1:
            print(Fore.RED + f'ERROR: extrude_1D() expects a 1D array!')
            return None
        column = vector.reshape((vector.shape[0], 1))
        return np.concatenate([column] * n_columns, axis=1)


    def min_max_norm(array: np.ndarray,
                    mins: np.ndarray,
                    maxes: np.ndarray,
                    bounds: tuple = (-1, 1)) -> np.ndarray:
        """
        Calculates the min-max normalized values of each row of an array given a compatible array of mins and maxes.
        This is used to min-max normalize each structural parameter (i.e. row) separately (post z-normalization).
        """
        
        return (bounds[1] - bounds[0]) * (array - mins) / (maxes - mins) + bounds[0]


    def normalize_profiles(profiles: np.ndarray, 
                        norm_paths: list = norm_data_path) -> np.ndarray:
        """
        Normalizes a profile based on the training set statistics (mean, sdev, post z-norm min, post z-norm max).
        """
        
        norm_params = [np.load(path)[:,0] for path in norm_paths]
        norm_params = [np.array([extrude_1D(param, profiles.shape[1])]) for param in norm_params]
        z = (profiles - norm_params[0][0]) / norm_params[1][0]
        normalized = min_max_norm(z, norm_params[2][0], norm_params[3][0])
        return normalized


    def decider_fx(probabilities: np.ndarray,
                thresholds: list = [0.7, 0.2, 0.2],
                priority_rank: list = np.array([1, 3, 2])) -> np.ndarray: # for priority_rank: higher = more priority
        """
        Calls positive (exon start or end) or negative (neither) positions in a prediction array based on class-specific model probability thresholds.
        """
        
        thresholded_output = np.zeros(shape=probabilities.shape[0]) 
        this_sample = np.zeros(shape=probabilities.shape)
        
        for j_class in range(probabilities.shape[1]):
            this_sample[:, j_class] = probabilities[:, j_class] > thresholds[j_class]
            
        for k_position in range(probabilities.shape[0]):
            in_the_running = np.argwhere(this_sample[k_position,:] == 1).flatten().tolist()
            if len(in_the_running) > 1:
                for cls in in_the_running:
                    if priority_rank[cls] == max(priority_rank[in_the_running]):
                        thresholded_output[k_position] = in_the_running[0]
            elif in_the_running == []:
                continue
            else:
                thresholded_output[k_position] = in_the_running[0]
        
        return thresholded_output


    def split_n_predict(profile: np.ndarray, 
                        model,
                        window_size: int = model_input_size[0], 
                        verbose: int = 0,
                        batch_size: int = 1024) -> tuple: 
        """
        Splits a structural profile into windows that the model can process and aggregates all predictions (each window is offset by 1).
        Uses Keras timeseries_dataset_from_array() for the windowing (adds efficiency and robustness to long sequence lengths).
        
        Args:
            profile: A Numpy array with the structural profile for a sequence, (of the shape (28 x [seq_length - unprocessed start/end windows]) and produced by 'profile_generator_inference.py').
            model: A Keras model object with weights fully loaded, must be one of the three trained during the work described in the pre-print.
            window_size: The model's context/input window, locked to the training window (here 77 bp).
            verbose: Whether or not Keras prints updates during inference (keep off, this slows things down).
            batch_size: Determines how many sequence windows (of window_size) are fed to the model for inference at once. This is important for very long sequences to not hang/run out of memory.
            
        Returns:
            A tuple with the raw model boundary-level probabilities for each position as the first element and the corresponding threshold-filtered boundary-level class calls as the second element.
        """
        
        dataset = timeseries_dataset_from_array(data=profile,
                                                targets=None,
                                                sequence_length=window_size,
                                                sequence_stride=1,
                                                shuffle=False,
                                                batch_size=batch_size)
        
        predicted = model.predict(dataset, verbose=verbose)
        decided = decider_fx(predicted) 
        
        return predicted, decided


    def length_prob(x: int, 
                    data: pd.DataFrame = length_dist) -> pd.DataFrame:
        """
        Looks up the probability of a length of an exon based on an approximate human distribution estimated from Mokry et al. (2010), see pre-print.
        """
        
        return np.interp(x, data.Length, data.Proportion)


    def make_exon_predictions(decisions: np.ndarray, 
                            probability: np.ndarray, 
                            max_exon_length: int = 997) -> list:
        """
        Takes the boundary-level predictions (class calls) and considers all possible resulting exons, finally filtering that down to a more limited set.
        1) Calculates exon-level scores based on the corresponding beginning and start position probabilities (for the called class) from the model.
        2) Reweighs the exon-level prediction's scores based on the rough probability of that hypothetical exon's length in the human genome.
        NOTE: This approach is a crude stand in for more complex model(s) that have also effectively learned the relevant gene structural/length distributions (see the pre-print)
        
        Args:
            decisions: A Numpy array with the boundary-level class calls from the model for each position in a sequence (one class assigned per position)
            probability: A Numpy array with the corresponding model probabilities (3 per position) from which those classes were called
            max_exon_length: This function will not pass and exon prediction longer than max_exon_length (where the length distribution estimated from Mokry et al. ended), this is a major limitation (see the pre-print)!
        
        Returns:
            A list of tuples with the start and end positions for the passed exons, as well as their length-adjusted exon-level score
        """
        
        start_class = 1
        end_class = 2
        exons = []
        
        starts = np.argwhere(decisions==start_class)
        ends = np.argwhere(decisions==end_class)
        these_exons = []
        if starts.size > 0:
            for start_idx in starts:
                if ends.size > 0:
                    for end_idx in ends:
                        if end_idx > start_idx and abs(end_idx - start_idx) < max_exon_length: 
                            combined_prob = (probability[start_idx, start_class] + probability[end_idx, end_class]) / 2
                            rescaled_prob = combined_prob/np.abs(np.log10(length_prob(end_idx - start_idx)))
                            these_exons.append((int(start_idx[0]), int(end_idx[0]), float(rescaled_prob[0])))
        exons.append(these_exons)
        
        return exons


    def greedy_filter_intervals_by_prob(series_list: list) -> list:
        """
        Filters a list of intervals/tuples (i.e. exon-level predictions), keeping only the highest-probability interval within any overlapping set.
        This is an imperfect way of enforcing the simple rule that overlapping exon predictions are not allowed.
        """
        
        sorted_list = sorted(series_list, key=lambda x: x[2], reverse=True)
        result_list = []
        
        def check_overlap(new_interval, existing_intervals):
            new_start, new_end, _ = new_interval
            for existing_start, existing_end, _ in existing_intervals:
                if new_start <= existing_end and new_end >= existing_start:
                    return True
            return False
        
        for current_series in sorted_list:
            if not check_overlap(current_series, result_list):
                result_list.append(current_series)
                
        return result_list


    def revise_and_filter(exons_source: list,
                        filter_threshold: int = None,
                        global_start_position: int = 0,
                        final_predictions_offset: int = final_predictions_offset) -> list:
        """
        Refines exon predictions by first filtering overlaps (see greedy_filter_intervals_by_prob() ) and then applying an exon-level probability threshold.
        """
        
        revised_exons = []
        for i in range(len(exons_source)):
            revised_exons.append(greedy_filter_intervals_by_prob(exons_source[i]))
        
        if filter_threshold is not None:
            filtered_revised = []
            for i in range(len(revised_exons)):
                this_seq_hits = []
                for j in range(len(revised_exons[i])):
                    if revised_exons[i][j][-1] > filter_threshold:
                        capped = (revised_exons[i][j][0] + final_predictions_offset + global_start_position, 
                                revised_exons[i][j][1] + final_predictions_offset + global_start_position, 
                                min(revised_exons[i][j][2], 0.99)) 
                        this_seq_hits.append(capped)
                filtered_revised.append(this_seq_hits)
            
            return filtered_revised[0]
        else:
            return revised_exons[0]


    def write_exons_to_gff(gff_path: str,
                        sequence_id: str,
                        features_list: list,
                        sequence_length: int = None,
                        source: str = 'Helix_EXon_finder',
                        feature_type: str = 'exon'):
        """
        Writes a list of predicted exon features to a GFF file.
        """
        
        newlines = []
        if features_list:
            for feature in features_list:
                attribute_string = f'CONFIDENCE_SCORE={round(feature[2], 6)}'
                if sequence_length is not None:
                    attribute_string = attribute_string + f';SEQUENCE_LENGTH={sequence_length}'
                newlines.append(f'{sequence_id}\t{source}\t{feature_type}\t{feature[0]}\t{feature[1]}\t{round(feature[2], 2)}\t.\t.\t{attribute_string}\n')
        else:
            # Use the generic template defined at the top, so visualize_predictions.py knows how to look to see if none were made.
            this_seq_specifics = {'sequence_id' : sequence_id, 'sequence_length' : sequence_length}
            newlines.append(NO_PREDICTIONS_TEMPLATE.format(**this_seq_specifics))
            
            # Log the seq_id of the sequence for which no predictions were made
            with open(no_predictions_log, mode = 'a') as log_file:
                log_file.writelines([f'{sequence_id}\n'])
        
        with open(gff_path, mode='w') as file:
            file.writelines(newlines)


    def pad_message(msg: str, 
                    length: int) -> str:
        """
        Pads a message with spaces to a desired length. Useful for properly overwriting console lines.
        """
        
        if len(msg) < length:
            msg += ' ' * (length - len(msg))
        return msg


    def get_bin_index(length: int,
                        bin_size: int = bin_size) -> int:
        """
        Determines the bin index for a given sequence length. Used to sort sequences into length-based bins for ETC calculations.
        """
        
        return int(length // bin_size)



    ### MAIN PROCEDURE
    ### MAKE EXON PREDICTIONS FOR ALL SEQUENCES IN TARGET FASTA FILE
    
    ## 1) Generate structural profiles for all valid sequences in the input FASTA
    shutil.rmtree(profiles_dir)
    profiles_dir.mkdir()
    full_msg = f"Generating profiles for sequences in '{input_fasta}'..."
    msg_length = len(full_msg)
    print(Fore.MAGENTA + full_msg, end='\r', flush=True)
    seqs_unpacked = pgi.generate_profiles(input_fasta,
                                          verbose = False,
                                          one_file_per_seq=True)
    seq_lengths = {seq_item[1]: seq_item[2] for seq_item in seqs_unpacked}
    full_msg = f"Finished generating profiles for sequences in '{input_fasta}'."
    full_msg = pad_message(full_msg, msg_length)
    print(Fore.GREEN + full_msg)
    if pgi.log_file_name in os.listdir(profiles_dir):
        print(Fore.YELLOW + f"WARNING: One or more sequences were skipped during profile generation, see '{shorten_path(profiles_dir / pgi.log_file_name, 2)}' for details.")
    print(Fore.WHITE + '---- TENSORFLOW OUTPUT (can likely ignore) ----')  # Reset color, prompt user to ignore TF output
    
    
    ## 2) Load profiles one by one and tally the sequence lengths
    profiles_list = list(profiles_dir.glob('*.npy'))
    profile_lengths = {}
    for i, profile_file in enumerate(profiles_list):
        profile = np.load(profile_file, mmap_mode='r')
        profile_lengths[profile_file.name] = profile.shape[1]
    total_bp = sum(profile_lengths.values())
    
    
    ## 3) Load the trained model, make predictions, and write results
    
    # Clear out predictions directory if any old prediction files exist
    if list(predictions_dir.glob('*')):
        shutil.rmtree(predictions_dir)
        predictions_dir.mkdir()
    
    # Load the trained model
    model = prepare_model(model_name=model_name) 
    
    # Initialize tracking variables
    msg_length = 0
    beginning = time.time() # TODO: move this to the beginning to include profile generation (report total pipeline time)?
    bin_stats = {} 
    total_processed_time = 0.0
    total_processed_bp = 0
    
    # Track remaining files to calculate ETC
    remaining_profiles = set(profiles_list)
    
    for i, profile_file in enumerate(profiles_list):
        
        # ETC Calculation Block
        if i > 0:
            estimated_remaining_time = 0.0
            
            # Calculate global rate (time per bp) as a fallback for empty bins
            global_rate_per_bp = total_processed_time / total_processed_bp if total_processed_bp > 0 else 0
            
            for remaining_file in remaining_profiles:
                rem_len = profile_lengths[remaining_file.name]
                rem_bin = get_bin_index(rem_len)
                
                if rem_bin in bin_stats and bin_stats[rem_bin][1] > 0:
                    # We have data for this length bin, use the specific average
                    avg_bin_time = bin_stats[rem_bin][0] / bin_stats[rem_bin][1]
                    estimated_remaining_time += avg_bin_time
                else:
                    # No data for this bin yet, use the global rate per bp instead
                    estimated_remaining_time += rem_len * global_rate_per_bp
            
            # Formatting the time string for status message
            if estimated_remaining_time > 60:
                msg = str(round(estimated_remaining_time / 60, 2)) + ' minutes'
            elif estimated_remaining_time > 3600:
                msg = str(round(estimated_remaining_time / 3600, 2)) + ' hours'
            else:
                msg = str(int(estimated_remaining_time)) + ' seconds'
        else:
            msg = 'calculating...'
        
        # Update Status Message
        full_msg = f'Processing profile {i+1} of {len(profiles_list)}: {profile_file.name}. Rough ETC: {msg}'
        full_msg = pad_message(full_msg, msg_length)
        if i == 0:
            print(Fore.GREEN + 'Stand by for inference update with an ETC. First update can take a minute or two and always overestimates (ETC accuracy improves).')
            print(Fore.WHITE + '---- TENSORFLOW OUTPUT (can likely ignore) ----')  # Reset color, prompt user to ignore TF output
        else:
            print(Fore.MAGENTA + full_msg, end='\r', flush=True)
        msg_length = len(full_msg)
        
        # Record start time
        start = time.time()
        
        # Remove current file from remaining set so it isn't counted in next ETC
        remaining_profiles.remove(profile_file)
        
        # Sequence ID is taken straight from filename (originally from FASTA headers)
        seq_id = profile_file.stem
        
        # 3.2) Load and normalize the profile
        profile = np.load(profile_file, mmap_mode='r') 
        normalized_profile = normalize_profiles(profile)
        
        # 3.3) Make predictions for the profile
        (probabilities, decisions) = split_n_predict(normalized_profile.transpose(), model=model)
        predicted_exons = make_exon_predictions(decisions, probabilities)
        revised_exons = revise_and_filter(predicted_exons, filter_threshold=exon_level_threshold)
        if len(revised_exons) != 0:
            revised_exons = sorted(revised_exons, key=lambda x: x[0])
        
        # 3.4) Write final predictions to a GFF
        gff_path = predictions_dir / f'{seq_id}.gff'
        write_exons_to_gff(gff_path=gff_path, 
                           sequence_length=seq_lengths[seq_id],
                           sequence_id=seq_id,
                           features_list=revised_exons)
        
        # Update ETC Stats
        end = time.time()
        duration = end - start
        current_len = profile_lengths[profile_file.name]
        current_bin = get_bin_index(current_len)
        
        # Update Bin Stats
        if current_bin not in bin_stats:
            bin_stats[current_bin] = [0.0, 0]
        bin_stats[current_bin][0] += duration
        bin_stats[current_bin][1] += 1
        
        # Update Global Stats
        total_processed_time += duration
        total_processed_bp += current_len
        
    # Show final completion message
    final_end = time.time()
    full_msg = f'HEX-finder finished making predictions for all {len(profiles_list)} sequences! Took {round((final_end - beginning)/60, 2)} minutes for {total_bp} bp of total sequence length.'
    full_msg = pad_message(full_msg, msg_length)
    print(Fore.GREEN + full_msg)
    
    if delete_profiles_after:
        if pgi.log_file_name in os.listdir(profiles_dir):
            shutil.copy2(profiles_dir / pgi.log_file_name, predictions_dir / pgi.log_file_name)
            print(Fore.GREEN + f"The log of skipped sequences was moved to '{shorten_path(predictions_dir / pgi.log_file_name, 2)}' in anticipation of profile deletion.")
        shutil.rmtree(profiles_dir)
        profiles_dir.mkdir()
        print(Fore.GREEN + f"Structural profiles in '{shorten_path(profiles_dir, 1)}' have been deleted as per user request.")
    else:
        print(Fore.GREEN + f"--> Structural profiles can be found in '{shorten_path(profiles_dir, 1)}' as NPY files (one file per processed input sequence).")
    
    print(Fore.GREEN + f"--> Exon-level predictions can be found in '{shorten_path(predictions_dir, 1)}' within GFF files (one file per processed input sequence).")
    print(Fore.YELLOW + "--> Consider using 'HEX-finder visualize' to look at the exons and compare them to known truth features (if available).")
    print(Fore.YELLOW + "--> Run 'HEX-finder visualize --help' for more information on this tool's usage.") #implemented 