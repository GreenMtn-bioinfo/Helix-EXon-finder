# Helix-EXon-finder

**H**elix-**EX**on-finder, or **HEX**-finder, is a command-line tool that uses a deep learning model trained on physicochemical profiles derived from DNA sequences for *ab initio* genomic feature detection:

+ The models and algorithms at the heart of this tool were developed, trained, and evaluated for my final project in the Brandeis GPS Bioinformatics program during an 8-week independence research course (RBIF 120). Please see the [pre-print](https://doi.org/10.64898/2025.12.19.694709) written about that work for more details.

+ This work was inspired by and based on work [published by Mishra et al. in 2021](https://doi.org/10.1093/nar/gkab098), as well as follow-up work published by [Sharma et al. in 2025](https://doi.org/10.1039/D4MO00241E).

+ The results in the [pre-print](https://doi.org/10.64898/2025.12.19.694709) make it clear that, in its current state, this tool and its underlying models are **not** competitive with current state-of-the-art *ab initio* gene detection tools. This is unsurprising, given that these models were trained on data that only encapsulates one of many aspect of gene's layout. That said, incorporation of this type of physicochemical/structural information into a more complex and refined approach may have potential. I am mainly providing the source code to document my effort, or in case anyone finds this idea promising and wants to build on it. I also hope to keep improving this tool past the performance demonstrated in the pre-print. For example, an HMM-based post-processing model coupled with the existing structural classification model(s) could result in an improvement in performance by prioritizing sets of hypothetical exon predictions using learned information about exon/intron length distributions in one or more vertebrate genomes.

+ Please contact **jhgmbioinfo@gmail.com** with any questions about the [pre-print](https://doi.org/10.64898/2025.12.19.694709) or source code.

+ Documentation on how to set up and use this repo is coming very soon.

+ A separate repository will be made to share the code written to train and evaluate the models at the heart of HEX-finder, which was the bulk of the work detailed in the [pre-print](https://doi.org/10.64898/2025.12.19.694709). This will come with no guarantee of portability/reproducibility, though basic documentation and a YAML to recreate the environment/dependencies will be provided. 