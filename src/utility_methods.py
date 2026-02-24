from colorama import Fore, init
init(autoreset=True)



### This module contains functions/methods used by more than one of the other scripts
### It is not very large right now but may save lines of code as the project grows
# TODO: Rename and refactor to also hold project-wide paths and constants in one place



def import_gff(gff_path: str,
               field_delim: str = '\t',
               line_delim: str = '\n') -> list:
    """
    Imports features from a GFF file (given its path) as a list of lists.
    
    Args:
        gff_path: Path to a standard GFF-formatted text file.
        field_delim: Character to use to split each line/string to determine the columns/fields.
        line_delim: Newline character that needs to be stripped.
        
    Returns:
        A list of lists: the outer list is the lines/rows, the inner lists are the columns/fields
    """
    
    features = []
    try:
        with open(gff_path, mode='r') as file:
            lines = file.readlines()
            if lines:
                for line in lines:
                    fields = line.strip(line_delim).split(field_delim)
                    features.append(fields)
                return features
            
            else:
                return None
            
    except FileNotFoundError:
        print(Fore.RED + f"ERROR: Input file not found at {gff_path}")
        
    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}")


def check_command_exit(result, 
                       msg: str,
                       stop_after_msg: bool = True):
    """
    Checks the exit status of shell command run via subprocess and ends the script (optional) with a message if it failed.
    """
    
    if result.returncode != 0:
        print(msg)
        if stop_after_msg:
            exit()