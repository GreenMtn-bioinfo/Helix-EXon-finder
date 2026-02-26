


### This is simple command line tool that retrieves sequences and relevant truth features from GRCh38.p14 in order to try HEX-finder.
### The user can provide their own coordinates to use, or sequences can be randomly sampled from the held-out regions.
### Warnings will be issued if the user provides coordinates that fall within the training regions of the genome (see ../data/held_out_regions_GRCh38_p14.txt).
### As an example, demo files and this script's corresponding output are provided for the sequences in Figure 11 of the manuscript (in ../demo_sequences/).
### This script does not have to be re-run unless you want to retrieve new sequences from GRCh38.p14 (randomly sampled or chosen by you).
### Also, this script is not necessary at all for running HEX-finder if you have your own sequence file (and/or local truth features file already, see README).
### This script must be run from a Linux command line environment with bedtools and samtools installed as well as Python and
### several Python libraries (see the conda config file provided in ../dependencies/)



def main():
    
    import argparse
    import sys
    from .paths import DEMO_SEQS_DIR, REFERENCE_DIR, HELD_OUT, FIG_11_DEMO, EXAMPLE_DIST, GENOME_FETCHER, shorten_path
    
    
    
    ### PARSE USER ARGUMENTS PROVIDED AT THE COMMAND LINE 
    #TODO: Improve formatting of this help description string
    parser = argparse.ArgumentParser(description="Facilitates retrieving sequences and reference features from GRCh38.p14 in order to try out HEX-finder's exon prediction capabilities. It will fetch the required reference genome if none is present or provided (see <reference_genome_directory>). This tool has two mutually exclusive use cases: 1) The user provides a path to a file of genomic coordinates (see <coordinates_file>) and sequences plus annotation are prepared for that set. 2) The user provides an integer (see <number_to_sample>), as well as a file with the allowed sequence lengths (see <lengths_distribution_file>), for random sampling. Coordinates from within the genomic regions withheld from the models' training set are first randomly sampled, and then the corresponding sequences and reference features are prepared. In either use case, reference exons within the sequences/coordinates of interest are retrieved and converted into local, 1-based coordinates (wrt to each sequence's start). The resulting GFF can be used with 'visualize_predictions.py' to visually evaluate HEX-finder's predictions directly against RefSeq MANE Select exons as a truth source.")

    # Arguments that apply to either use case
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='If specified, prevents non-critical script status updates from being printed to the console.')
    parser.add_argument('-r','--reference_dir', type=str, default=REFERENCE_DIR, metavar='<reference_genome_directory>',
                        help=f'Path to a directory including an indexed reference genome file (FNA and FAI) for GRCh38.p14, along with a RefSeq annotation file (GFF). The default is "{shorten_path(REFERENCE_DIR, 2)}". If any of the required files are not found, the user will be prompted to give permission to retrieve and prepare them automatically.')

    # Create a mutually exclusive group for the two use cases
    use_case = parser.add_mutually_exclusive_group(required=True)

    # OPTION 1: User provides a file with valid genomic coordinates, given the reference genome
    # user_provides_coords = parser.add_argument_group(title='User provides coordinates for sequence retrieval.')
    use_case.add_argument('-c','--coordinates', type=str, default=None, metavar='<coordinates_file>',
                        help=f'Path to a file including one genomic coordinate per line. Coordinate format should be CHROMOSOME:START-END(STRAND_SYMBOL), please see "{shorten_path(FIG_11_DEMO, 2)}" for a valid example.')

    # OPTION 2: User provides a number of coordinates to randomly sample from the held-out region and a file with sequence lengths to draw from
    use_case.add_argument('-n', '--number', type=int, default=None, metavar='<number_to_sample>',
                        help='An integer specifying how many coordinates to randomly sample from the regions of GRCh38.p14 that were held-out during model training.')
    sampling_group = parser.add_argument_group(title='used only when <number_to_sample> is provided')
    sampling_group.add_argument('-l', '--lengths', type=str, default=None, metavar='<lengths_distribution_file>',
                                help=f'Path to a file with one sequence length per line (integers). This will be used as the distribution of possible lengths for the random coordinate sampling. Please see or use "{shorten_path(EXAMPLE_DIST, 2)}" as a valid example.')
    sampling_group.add_argument('-o', '--output', type=str, default=None, metavar='<output_base_name>',
                                help=f'The base name to use for all output files if random sampling is chosen (defaults to "<number_to_sample>_seqs"). All files will be saved in "{shorten_path(DEMO_SEQS_DIR, 1)}".')
    sampling_group.add_argument('-s', '--seed', type=int, default=2026, metavar='<seed_integer>',
                                help='An integer seed to use for the random coordinate sampling (defaults to 2026). Leave this alone for reproducible sampling.')

    # Parse arguments and check if help was called
    args = parser.parse_args()
    if '--help' in sys.argv or '-h' in sys.argv:
        sys.exit(0)

    # Import more modules now that we know it's not just a help call
    from .utility_methods import import_gff, check_command_exit
    import subprocess
    import numpy as np
    import os
    import re
    import shutil
    from colorama import Fore, init
    init(autoreset=True)

    # Assign user arguments received from the command line
    seq_coords_file = args.coordinates
    reference_dir = args.reference_dir
    n_regions = args.number
    lengths_file = args.lengths
    output_name = args.output if args.output else f'{n_regions}_seqs'
    quiet = args.quiet
    seed = args.seed
    rng = np.random.default_rng(seed=seed) # For reproducibility, if desired



    ### ESTABLISH PATHS USED THROUGHOUT THE SCRIPT (DO NOT CHANGE THESE)

    # Where the output files are written to
    operating_dir = DEMO_SEQS_DIR

    # Relative path to the file that contains the regions of GRCh38.p14 omitted from the models' training data
    held_out_file_path = HELD_OUT

    # Define paths of temporary files that are created as necessary intermediates and then deleted
    temp_dir = DEMO_SEQS_DIR / 'temp'
    temp_dir.mkdir(exist_ok=True)
    temp_region_path = temp_dir / 'temp_region.txt'
    temp_gff_path = temp_dir / 'temp_coords.gff'

    # This is used to write output if random sampling was chosen
    sampled_coords_path = operating_dir / f'{output_name}_coords.txt'

    # This is used to write sequence output file
    output_fasta = operating_dir / f'{output_name}.fasta'

    # This is used to write truth features output file
    truth_features_path = operating_dir / f'{output_name}_truth_features_local.gff'

    # Define paths for temporary GFF files used during processing
    mane_select_exons_gff = temp_dir / 'mane_select_exons.gff'
    mane_select_forward_exons_gff = temp_dir / 'mane_select_forward_exons.gff'
    mane_select_reverse_exons_gff = temp_dir / 'mane_select_reverse_exons.gff'



    ### FUNCTION DEFINITIONS

    def check_ref_dir(reference_dir: str) -> tuple:
        """
        Verifies/establishes paths to an indexed reference genome and annotation.
        
        Args:
            reference_dir: path to a directory that contains compatible FNA, FAI, and GFF files.
        
        Returns:
            A tuple with confirmed the paths to the: FNA, FAI, and GFF files.
        """
        
        reference_files = os.listdir(reference_dir)
        reference_fasta_path, reference_fasta_index, annotation_gff_path = None, None, None
        for file_name in reference_files:
            suffix = file_name[-4:]
            match suffix:
                case '.fna':
                    reference_fasta_path = reference_dir / file_name
                case '.fai':
                    reference_fasta_index = reference_dir / file_name
                case '.gff':
                    annotation_gff_path = reference_dir / file_name
        return reference_fasta_path, reference_fasta_index, annotation_gff_path


    def make_temp_gff(chrom: str, # or chromosome
                    start: int,
                    end: int,
                    strand: str, # - or +
                    temp_gff_path: str = temp_gff_path):
        """
        Creates a temporary, one-line GFF from provided pieces of a genomic coordinate (chrom, start_pos, end_pos, and strand symbol)
        Such a file is necessary to call bedtools intersect, as it expects regions/features to be in GFF format or similar.
        """
        
        with open(temp_gff_path, mode='w') as file:
            file.write(f'{chrom}\tarbitrary_selection\tregion\t{start}\t{end}\t.\t{strand}\t.\t.\n')


    def BED_intersect(ref_features_path: str, # truth_gff
                    range_path: str, # gff with a range or set of ranges of interest
                    output_path: str) -> subprocess.CompletedProcess:
        """
        Bedtools intersect wrapper. Calls bedtools to find the reference features that overlap with a genomic coordinate.
        
        Args:
            ref_features_path: Path to the reference GFF file containing features to intersect.
            range_path: Path to a GFF file specifying genomic region(s) of interest (here only ever one).
            output_path: Path to save the result of bedtools intersect.
        
        Returns:
            An object with the details/results of the shell execution (mainly used to check exit status).
        """
        
        if '.gff' not in str(ref_features_path) or '.gff' not in str(range_path):
            print(Fore.RED + 'ERROR: BED_intersect() requires the paths to two GFF files (.gff) as input!')
            return
        else:
            command = f'bedtools intersect -a "{ref_features_path}" -b "{range_path}" > "{output_path}"'
            boundary_seqs = subprocess.run(command, shell=True)
            return boundary_seqs


    def check_and_delete(paths_of_interest: list):
        """
        Checks if each in a list of paths (files or directories) exists and deletes them if they do.
        """
        
        for path in paths_of_interest:
            if os.path.exists(path):
                os.remove(path)


    def samtools_faidx_wrapper(region_file_path: str,
                            reverse_strand: bool,
                            output_path: str,
                            reference_fasta_path,
                            max_line_length: int = 75) -> subprocess.CompletedProcess:
        """
        Basic wrapper around the samtools faidx command line tool.
        Used to fetch sequences from the reference genome given a file of genomic coordinates.
        
        Args:
            region_file_path: Path to a file containing genomic coordinates in the format 'chromosome:start_position:end_position' (one per line). The file should be formatted as per the samtools faidx documentation.
            reverse_strand: A boolean indicating whether the reverse complement sequence is to be pulled for the coordinates.
            output_path: Path to a FASTA file to which the retrieved sequence will be appended.
            reference_fasta_path: Path to the FNA/FASTA file of the reference genome/source of the sequence.
            max_line_length: What is the line max length to use when wrapping the sequences in the output file?
        
        Returns:
            An object with the details/results of the shell execution (mainly used to check exit status here).
        """
        
        # Fetch this sequence from the reference FASTA and append to output file
        strand_flag = '-i' if reverse_strand else ''
        command = f'samtools faidx -n {max_line_length} {reference_fasta_path} -r {region_file_path} {strand_flag} --mark-strand sign >> {output_path}'
        execute_status = subprocess.run(command, shell=True)
        return execute_status


    def convert_coords_to_local(seq_start_global: int,
                                seq_end_global: int,
                                feature_coords: tuple,
                                reverse_strand: bool) -> tuple:
        """
        Converts global sequence coordinates (within a chromosome or large scaffold) to local coordinates based on the global start position of the sequence of interest.
        
        Args:
            seq_start_global: The global start position of the sequence of interest.
            feature_coords: A tuple representing the global coordinates (start, end) to convert.
        Returns:
            A tuple representing the local 1-based coordinates (start, end), i.e. relative to the start of the sequence of interest.
        """
        
        start = feature_coords[0]
        end = feature_coords[1]
        seq_length = seq_end_global - seq_start_global + 1
        
        # First convert to the local, i.e. bp of offset/position relative to global start
        new_start = start - seq_start_global + 1
        new_end = end - seq_start_global + 1
        
        # If the sequence was pulled from the reverse strand we need to modify the coordinates to be relative to the end (coords are + strand oriented)
        if reverse_strand: 
            corrected_start = seq_length - new_end + 1
            corrected_end = seq_length - new_start + 1
            new_start, new_end = corrected_start, corrected_end
        
        return (new_start, new_end)


    def fetch_truth_features(chrom: str,
                            coord_start: int,
                            coord_end: int,
                            strand_symbol: str,
                            truth_annotation_path: str,
                            output_path: str):
        """
        Finds the overlapping truth/reference features for a given genomic region and writes them to a GFF file (using bedtools).
        Importantly, the coordinates of these overlapping features are saved as local 1-based coordinates (i.e. wrt the sequence start/length).
        This facilitates direct comparison to HEX-finder predictions (e.g. using 'visualize_predictions.py'), which are wrt the sequence.
        
        Args:
            chrom: The chromsome identifier for the reference genome being used
            coord_start: The start position of the genomic region.
            coord_end: The end position of the genomic region.
            strand_symbol: The strand symbol, wrt to the reference genome ('+' or '-').
            truth_annotation_path: Path to the reference annotation GFF file.
            output_path: Where to save the overlapping features.
        """
        
        # Initialize variables based on user input
        seq_id = f'{chrom}:{coord_start}-{coord_end}({strand_symbol})'
        truth_temp_file_path = temp_dir / f'{seq_id}_truth_features.gff'
        
        # Find the truth features that overlap with the current region
        make_temp_gff(chrom=chrom,
                    start=coord_start,
                    end=coord_end, 
                    strand = strand_symbol)
        result = BED_intersect(ref_features_path=truth_annotation_path,
                            range_path=temp_gff_path,
                            output_path=truth_temp_file_path)
        check_command_exit(result,
                        Fore.RED + f"ERROR: Failed to use bedtools intersect to get truth features for coordinate {seq_id}. Is bedtools installed and was it fed valid coordinate files?")
        features = import_gff(truth_temp_file_path)
        
        # Convert global coordinates to local coordinates and append all to the output file
        local_coord_features = []
        if features:
            for feature in features:
                source, feature_type = feature[1:3]
                start = int(feature[3])
                end = int(feature[4])
                score, _, phase, attributes = feature[5:]
                start, end = convert_coords_to_local(seq_start_global = coord_start, 
                                                    seq_end_global = coord_end,
                                                    feature_coords = (start, end), 
                                                    reverse_strand = strand_symbol == '-')
                local_coord_features.append(f'{seq_id}\t{source}\t{feature_type}\t{start}\t{end}\t{score}\t.\t{phase}\t{attributes}\n')
        
        # Save all truth features to one final file (for all regions)
        with open(output_path, mode='a') as file:
            file.writelines(local_coord_features)


    def parse_coords(coord_string: str):
        """
        Parses a genomic coordinate string to extract chromosome/contig identifier, start, end, and strand (if present).
        """
        
        expr = r'(\w+_\d+\.\d+):(\d+)-(\d+)(\([\+|-]\))*'
        match = re.match(expr, coord_string)
        return match


    def check_and_read(file_path: str):
        """
        Checks if a simple one-string-per-line file exists before reading it in.
        """
        
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                lines = [line.strip('\n') for line in lines]
            return lines
        else:
            print(Fore.RED + f"The file '{file_path}' does not appear to exist. Please check/correct this path and try again.")
            exit()
    
    
    
    ### MAIN PROCEDURE
    
    ### 1) PREPARATION STEPS (REFERENCE GENOME, HELD-OUT CHECK, ETC.)
    
    # Check for a reference genome. Warn/stop if any reference files are missing and ask to retrieve reference files if needed
    if not quiet:
        print(Fore.MAGENTA + 'Checking for a reference genome and annotation...')
    if not os.path.exists(reference_dir):
        os.makedirs(reference_dir)
    if any(not exists for exists in check_ref_dir(reference_dir)):
        print(Fore.RED + f"ERROR: One or more of the necessary reference files (.fna, .fai, and .gff) were not found in '{reference_dir}'.")
        print(Fore.YELLOW + f"--> Please specify a path to a different directory with your own copies of these 3 files using the -r or --reference_dir option.")
        print(Fore.YELLOW + "--> Alternatively, run 'fetch_reference.sh' to get an indexed reference genome (GRCh38.p14) and corresponding annotation.")
        fetch_reference = input("--> Would you like to run 'fetch_reference.sh' now to get GRCh38.p14 and the required annotation? (y/n) ")
        if fetch_reference == 'y':
            command = f"{GENOME_FETCHER} {REFERENCE_DIR}"
            execute_status = subprocess.run(command, shell=True)
            if any(not exists for exists in check_ref_dir(reference_dir)):
                print(Fore.RED + "Something went wrong during reference genome retrieval.\nPlease check your internet connection and/or the FTP addresses in 'fetch_reference.sh' and try again.")
                exit()
        else:
            exit()
    reference_fasta_path, reference_fasta_index, annotation_gff_path = check_ref_dir(reference_dir)
    if not quiet:
        print(Fore.GREEN + 'Found a reference genome and annotation.')
    
    # Read the file of held-out regions to facilitate sequence sampling or chosen coordinate checks/warnings
    with open(held_out_file_path, 'r') as file:
        lines = file.readlines()
        held_out_regions = [line.strip('\n').split('\t') for line in lines][1:] # skip header
    
    
    ### 2) READ OR SAMPLE A SET OF COORDINATES
    
    # Has a file of specific coordinates from GRCh38.14 been provided? 
    
    # If so, that file will be use to pull sequences and truth features.
    if seq_coords_file is not None: 
        
        # Import the file of coordinates provided by the user if it exists
        chosen_file = seq_coords_file
        if not quiet:
            print(Fore.GREEN + f"Proceeding with user-provided genomic coordinates in file '{chosen_file}'.")
    
    # Otherwise, coordinates and sequences will be randomly sampled from held-out regions.
    elif all((n_regions, lengths_file, output_name)): 
        
        # Check/Import the file of allowed sequence lengths provided by the user
        lengths_pool = check_and_read(lengths_file)
        lengths_pool = [int(length) for length in lengths_pool]
        
        # Randomly sample chromosomes and sequence lengths from those specified above
        chromosomes = rng.choice(held_out_regions, size=n_regions, replace=True)
        lengths = rng.choice(lengths_pool, size=n_regions, replace=True)
        sampled_strands = rng.choice(["-", "+"], size=n_regions, replace=True)

        # Randomly select a starting location for each sequence, making sure not to run into the end of the current held-out region
        chosen_regions = []
        for i, chromosome in enumerate(chromosomes):
            assembly = chromosome[0]
            start = int(chromosome[1])
            end = int(chromosome[2]) - lengths[i]
            coord_start = rng.choice(range(start, end, 1000), size=1)[0]
            chosen_regions.append(f'{assembly}:{coord_start}-{coord_start + lengths[i] - 1}({sampled_strands[i]})\n')

        # Write the chosen regions to file
        with open(sampled_coords_path, 'w') as file:
            file.writelines(chosen_regions)
        chosen_file = sampled_coords_path
        if not quiet:
            print(Fore.GREEN + f"Sampled {n_regions} coordinates from within held-out regions of the genome, with lengths drawn from '{lengths_file}'.")
    
    # Handle if the user does not provide the parameters to follow either path/use case
    else:
        print(Fore.RED + "ERROR: Insufficient arguments provided for random sampling. At a minimum, you must either specify:")
        print(Fore.RED + f"1) A path to a valid coordinates file (-c). The file '{shorten_path(FIG_11_DEMO, 2)}' has been provided as a valid example.")
        print(Fore.RED + "OR")
        print(Fore.RED + f"2) A number of coordinates to randomly sample (-n) and a file of allowed lengths (-l, see '{shorten_path(EXAMPLE_DIST, 2)}' for an example).")
        print(Fore.YELLOW + "Please see the README or re-run this tool with --help for more information.")
        exit()
    
    
    ### 3) PROCEED FETCHING SEQUENCES AND ANNOTATION FOR THE COORDINATES (WHETHER SAMPLED OR PROVIDED)
    chosen_regions = check_and_read(chosen_file)
    
    # Extract MANE select exon annotation for GRCh38 and separate by strand (used later to grab overlapping "truth features" for comparison to predictions)
    if not quiet:
        print(Fore.MAGENTA + 'Extracting relevant exon annotation from the reference...')
    command = f"cat {annotation_gff_path} | grep -P '\tBestRefSeq\t' | grep -P 'tag=MANE\\sSelect' | grep -P '\texon\t' | grep '^NC_' > {mane_select_exons_gff}"
    boundary_annotation = subprocess.run(command, shell=True)
    check_command_exit(boundary_annotation, 
                    Fore.RED + f"ERROR: Failed to prepare RefSeq MANE Select annotation from file '{annotation_gff_path}'.\nMake sure all dependencies are installed and this script is running from within a typical Linux Bash environment (see README).")
    for strand_symbol in ['-', '+']:
        strand = 'reverse' if strand_symbol == '-' else 'forward'
        command = f"cat {mane_select_exons_gff} | grep -P '\t\{strand_symbol}\t' > {temp_dir / f'mane_select_{strand}_exons.gff'}"
        strand_annotation = subprocess.run(command, shell=True)
        check_command_exit(strand_annotation,
                        Fore.RED + f"ERROR: Failed to prepare exon annotation for the {strand_symbol} strand in file '{annotation_gff_path}'.\nMake sure all dependencies are installed and this script is running from within a typical Linux Bash environment (see README).")
    if not quiet:
        print(Fore.GREEN + 'Extracted and prepared the relevant exon annotation.')
    
    # Delete any previous feature or FASTA files with the same names (new ones would be appended to the old file otherwise)
    check_and_delete([truth_features_path, output_fasta])
    
    # Use samtools faidx to retrieve the sequences from the reference FASTA, 
    if not quiet:
        print(Fore.MAGENTA + f"Retrieving the sequences and overlapping reference annotation for coordinates in '{chosen_file}'...")
    
    for i, region in enumerate(chosen_regions):
        
        # Parse coordinate string into the useful parts
        match = parse_coords(region)
        if match:
            chromosome = match.group(1)
            coord_start = int(match.group(2))
            coord_end = int(match.group(3))
        else:
            print(Fore.YELLOW + f"WARNING: The coordinate on line {i+1} of '{chosen_file}' is not formatted correctly, skipping it.")
            continue
        
        # Check that the coordinates fall in the regions held-out during training (issue warning if not)
        unseen_seq_flag = False
        for held_out in held_out_regions:
            if coord_start >= int(held_out[1]) and coord_start <= int(held_out[2]):
                unseen_seq_flag = True
        if not unseen_seq_flag:
            print(Fore.YELLOW + f"WARNING: {region} on line {i+1} of '{chosen_file}' overlaps with HEX-finder's training region, interpret model output/performance with this in mind!")
        
        # Handle strand symbol (+ or -), if found
        if match.group(4) is not None:
            strand_symbol = match.group(4).strip('()')
        else:
            print(Fore.YELLOW + f'WARNING: No strand information provided in coordinate "{region}", skipping it.')
            continue
        
        # Write a temporary file with the coordinate (strand symbol stripped so samtools faidx can use it)
        with open(temp_region_path, mode='w') as file:
            file.writelines([f'{chromosome}:{coord_start}-{coord_end}'])
        
        # Fetch this sequence from the reference FASTA and append to output file
        result = samtools_faidx_wrapper(region_file_path = temp_region_path,
                                        reverse_strand = strand_symbol == '-',
                                        output_path = output_fasta,
                                        reference_fasta_path = reference_fasta_path)
        check_command_exit(result,
                        Fore.RED + f"ERROR: Failed to use samtools to fetch the sequence for coordinate {region}, from line {i+1} of '{chosen_file}'.\nIs this a valid coordinate and is samtools installed in this environment?")
        
        ## Identify overlapping RefSeq features, and convert to LOCAL (sequence-relative) coordinates, and save
        fetch_truth_features(chrom = chromosome,
                            coord_start = coord_start,
                            coord_end = coord_end,
                            strand_symbol = strand_symbol,
                            truth_annotation_path = mane_select_forward_exons_gff if strand_symbol == '+' else mane_select_reverse_exons_gff,
                            output_path = truth_features_path)
    
    if not quiet:
        print(Fore.GREEN + f"Sequences successfully retrieved for all valid coordinates and written to '.{os.path.sep}{shorten_path(output_fasta, 2)}'.")
        print(Fore.GREEN + f"Overlapping exon annotation was retrieved, converted to local coordinates, and written to '.{os.path.sep}{shorten_path(truth_features_path, 2)}'.")
        with open(truth_features_path, mode='r') as file:
            lines = file.readlines()
            if not lines:
                print(Fore.GREEN + f"No reference exons were found for these sequences, but this GFF is still valid to use with 'visualize_predictions.sh'.")
    
    # Clear the temp folder, contents are no longer needed
    shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
    
    # Print final messages
    print(Fore.YELLOW + f"--> You can now run 'python HEX-finder.py -f .{os.path.sep}{shorten_path(output_fasta, 2)}' to make predictions.")
    print(Fore.YELLOW + f"--> Run 'python HEX-finder.py --help' for more information on the options (see the README).")



if __name__ == '__main__':
    main()