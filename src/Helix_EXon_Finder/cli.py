import argparse
from .get_demo_seqs import main as get_demo_seqs_main
from .hex_finder import main as hex_finder_main
from .visualize_predictions import main as visualize_predictions_main
from .paths import DEMO_SEQS_DIR, REFERENCE_DIR, FIG_11_DEMO, EXAMPLE_DIST, PREDICTIONS_DIR, VISUAL_PREDICTIONS, shorten_path



# TODO: add a robust check that this is being run with the right dependencies available, i.e. via:
# 1) from within the conda environment specified in provided environment.yml file
# 2) via test imports to make sure key packages are installed and available
# Direct the user to the install commands in the README if not



def main():
    
    parser = argparse.ArgumentParser(description="This is the command-line interface for using HEX-finder and its auxiliary tools.")
    subparsers = parser.add_subparsers(dest='command', help='Use -h/--help with any one of these for more details on usage:')
    
    
    
    #### Subparser for get_demo_seqs.py
    parser_get_demo_seqs = subparsers.add_parser('fetch', # TODO: Improve formatting of the description string
        help='Fetch sequences from GRCh38.p14 to try out HEX-finder, along with reference exons (for use with "visualize" after "predict").',
        description="Facilitates retrieving sequences and reference features from GRCh38.p14 in order to try out HEX-finder's exon prediction capabilities. It will fetch the required reference genome if none is present or provided (see <reference_genome_directory>). This tool has two mutually exclusive use cases: 1) The user provides a path to a file of genomic coordinates (see <coordinates_file>) and sequences plus annotation are prepared for that set. 2) The user provides an integer (see <number_to_sample>), as well as a file with the allowed sequence lengths (see <lengths_distribution_file>), for random sampling. Coordinates from within the genomic regions withheld from the models' training set are first randomly sampled, and then the corresponding sequences and reference features are prepared. In either use case, reference exons within the sequences/coordinates of interest are retrieved and converted into local, 1-based coordinates (wrt to each sequence's start). The resulting GFF can be used with 'visualize_predictions.py' to visually evaluate HEX-finder's predictions directly against RefSeq MANE Select exons as a truth source.")
    
    # Arguments that apply to either use case
    parser_get_demo_seqs.add_argument('-q', '--quiet', action='store_true',
                        help='If specified, prevents non-critical script status updates from being printed to the console.')
    parser_get_demo_seqs.add_argument('-r','--reference_dir', type=str, default=REFERENCE_DIR, metavar='<reference_genome_directory>',
                        help=f'Path to a directory including an indexed reference genome file (FNA and FAI) for GRCh38.p14, along with a RefSeq annotation file (GFF). The default is "{shorten_path(REFERENCE_DIR, 2)}". If any of the required files are not found, the user will be prompted to give permission to retrieve and prepare them automatically.')

    # Create a mutually exclusive group for the two use cases
    use_case = parser_get_demo_seqs.add_mutually_exclusive_group(required=True)

    # OPTION 1: User provides a file with valid genomic coordinates, given the reference genome
    # user_provides_coords = parser.add_argument_group(title='User provides coordinates for sequence retrieval.')
    use_case.add_argument('-c','--coordinates', type=str, default=None, metavar='<coordinates_file>',
                        help=f'Path to a file including one genomic coordinate per line. Coordinate format should be CHROMOSOME:START-END(STRAND_SYMBOL), please see "{shorten_path(FIG_11_DEMO, 2)}" for a valid example.')

    # OPTION 2: User provides a number of coordinates to randomly sample from the held-out region and a file with sequence lengths to draw from
    use_case.add_argument('-n', '--number', type=int, default=None, metavar='<number_to_sample>',
                        help='An integer specifying how many coordinates to randomly sample from the regions of GRCh38.p14 that were held-out during model training.')
    sampling_group = parser_get_demo_seqs.add_argument_group(title='used only when <number_to_sample> is provided')
    sampling_group.add_argument('-l', '--lengths', type=str, default=None, metavar='<lengths_distribution_file>',
                                help=f'Path to a file with one sequence length per line (integers). This will be used as the distribution of possible lengths for the random coordinate sampling. Please see or use "{shorten_path(EXAMPLE_DIST, 2)}" as a valid example.')
    sampling_group.add_argument('-o', '--output', type=str, default=None, metavar='<output_base_name>',
                                help=f'The base name to use for all output files if random sampling is chosen (defaults to "<number_to_sample>_seqs"). All files will be saved in "{shorten_path(DEMO_SEQS_DIR, 1)}".')
    sampling_group.add_argument('-s', '--seed', type=int, default=2026, metavar='<seed_integer>',
                                help='An integer seed to use for the random coordinate sampling (defaults to 2026). Leave this alone for reproducible sampling.')
    
    
    
    #### Subparser for hex_finder.py
    parser_hex_finder = subparsers.add_parser('predict',
        help='Use HEX-finder to make exon predictions for a set of sequences.',
        description="Helix-EXon-finder (HEX-finder): Predict exons from genomic DNA sequences using a deep learning network trained on predicted structural profiles. For more details on the underlying methods and performance, see the README and accompanying pre-print (https://doi.org/10.64898/2025.12.19.694709).")
    parser_hex_finder.add_argument('-f','--fasta', type=str, required=True, metavar='<path_to_fasta>',
                        help='Path to an input FASTA file containing genomic sequences to analyze. Sequence IDs, which are used as profile (NPY) and prediction (GFF) file names, are taken from the header between ">" and the next whitespace. Please keep your sequence IDs concise, unique, and filename friendly.')
    parser_hex_finder.add_argument('-m', '--model', type=str, default='TCN', choices=['TCN', 'BiLSTM', 'MBDA-Net'], metavar= "<'TCN' or 'BiLSTM' or 'MBDA-Net'>",
                        help='Choice of trained model to use for predictions. TCN is the fastest and recommended for inference. MBDA-Net is slower but may perform marginally better.')
    parser_hex_finder.add_argument('-t', '--threshold', type=float, default=0.75, metavar='<probability_threshold_float>',
                        help='Exon-level probability score threshold for reporting an exon prediction (default is 0.75). Please see the README and/or pre-print for guidance on threshold selection.')
    parser_hex_finder.add_argument('-d', action='store_true',
                        help='If specified, the structural profiles will be deleted after predictions are made to save disk space. For reference, the structural profiles for ~0.577 Gbp of sequence take up ~90 GB of disk space.')
    
    
    
    #### Subparser for visualize_predictions.py
    visualize_prediction = subparsers.add_parser('visualize',
        help="Generate plots of HEX-finder's predictions (alongside known features for the same sequences, if provided).",
        description="Creates a simple, self-contained HTML report for visualizing HEX-finder predictions and comparing against truth features (if available).")
    visualize_prediction.add_argument('-p','--predictions_dir', type=str, default=PREDICTIONS_DIR, metavar='<predictions_directory_path>',
                        help=f"Path to a directory containing GFFs with predictions generated by HEX-finder (defaults is '{shorten_path(PREDICTIONS_DIR, 1)}').")
    visualize_prediction.add_argument('-t','--truth_features', type=str, default=None, metavar='<path_to_local_coords_GFF>',
                        help=f"Path to a single GFF file containing the local 1-based coordinates (wrt sequence start) of known reference features for all of the sequences of interest. Sequence IDs in the GFF must match those pulled by HEX-finder from the FASTA headers. See the GFFs provided in '{shorten_path(DEMO_SEQS_DIR, 1)}' for valid formatting. While not necessary, the 9th column (i.e. attributes field) can contain annotation used to label individual reference features (see <feature_attribute_name>). If left unspecified, only the HEX-finder predictions will be plotted.")
    visualize_prediction.add_argument('-tl','--truth_labels_attribute', type=str, default=None, metavar='<feature_attribute_name>',
                        help=f"Name of the attribute in the attributes field/column of the truth GFF from which to pull individual feature labels (see <path_to_local_coords_GFF> above). Attributes field must use ';' for attribute separation and '=' for attribute assignment. See the GFFs provided in '{shorten_path(DEMO_SEQS_DIR, 1)}' for valid examples. For RefSeq annotation, 'gene' will label each feature with its HUGO gene symbol. If left unspecified, each reference feature will simply be labeled as what is provided for <reference_features_source>.")
    visualize_prediction.add_argument('-tn','--truth_source_name', type=str, default=None, metavar='<reference_features_source>',
                        help="Name of the source of reference annotation HEX-finder is being compared to (for the axis labels/legend, e.g. 'BestRefSeq'). If unspecified, the script attempts to pull this from the 'source' field of the GFF file of truth_features (if provided and non-empty). If a homogenous source name is not found in that file, the label defaults to 'Reference'.")
    visualize_prediction.add_argument('-o','--output_path', type=str, default=VISUAL_PREDICTIONS, metavar='<output_HTML_path>',
                        help=f"Path to write the final HTML report to (the default is '{shorten_path(VISUAL_PREDICTIONS, 2)}').")
    visualize_prediction.add_argument('-se','--skip_empty', action='store_true',
                        help="If specified, sequences for which both HEX-finder and the reference/truth source had no exon features will not be included in the report. By default, these sequences are still plotted/included but are blank).")
    visualize_prediction.add_argument('-v', '--verbose', action='store_false',
                        help='If specified, enables more status updates to be printed to the console (off by default). Strongly recommended to leave this unspecified if HEX-finder has made predictions for many sequences.')
    visualize_prediction.add_argument('-js', '--javascript_included', action='store_true',
                        help="If specified, Plotly's JavaScript source code is baked into the HTML report. This ensures fully functional plots when there is no internet connection. This increases the report size from <100 Kb to ~4.5 Mb.")
    visualize_prediction.add_argument('-ac', '--accessibility_colors', action='store_true',
                        help="If specified, an alternative color palette is used in the report plots that should offer an improvement over the default for those with most forms of color blindness.")
    
    
    
    #### Parse the arguments for the selected command and run the right script (or show help)
    args = parser.parse_args()

    if args.command == 'fetch':
        get_demo_seqs_main(args)
        
    elif args.command == 'predict':
        hex_finder_main(args)
        
    elif args.command == 'visualize':
        visualize_predictions_main(args)
        
    else:
        parser.print_help()



if __name__ == '__main__':
    main()