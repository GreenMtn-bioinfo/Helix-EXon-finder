import argparse
from .get_demo_seqs import main as get_demo_seqs_main
# from src.hex_finder import main as hex_finder_main
# from src.visualize_predictions import main as visualize_predictions_main
from .paths import DEMO_SEQS_DIR, REFERENCE_DIR, FIG_11_DEMO, EXAMPLE_DIST, shorten_path



def main():
    parser = argparse.ArgumentParser(description="This is the main entry point for using HEX-finder and its auxiliary tools.")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command details:')


    #### Subparser for get_demo_seqs
    parser_get_demo_seqs = subparsers.add_parser( # TODO: Improve formatting of the description string
        'fetch_seqs',
        help='Fetch sequences from GRCh38.p14 to try out HEX-finder. Use -h/--help with this sub-command for more details.',
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


    # Add other subparsers similarly
    # parser_another_tool = subparsers.add_parser('another_tool', help='Description of another tool')
    # parser_another_tool.add_argument(...)
    

    #### Parse the arguments for the selected command and run the right script (or show help)
    args = parser.parse_args()

    if args.command == 'fetch_seqs':
        get_demo_seqs_main(args)
    # elif args.command == 'another_tool':
    #     another_tool_main(args)
    else:
        parser.print_help()



if __name__ == '__main__':
    main()