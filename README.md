# Helix-EXon-finder

**H**elix-**EX**on-finder, or **HEX**-finder, is a command-line tool that uses a deep learning model trained on physicochemical profiles derived from DNA sequences for *ab initio* genomic feature detection:

+ The models and algorithms at the heart of this tool were developed, trained, and evaluated for my final project in the Brandeis GPS Bioinformatics program for an independence research course (RBIF 120).

+ This work was inspired by and based on work [published by Mishra et al. in 2021](https://doi.org/10.1093/nar/gkab098), as well as follow-up work published by [Sharma et al. in 2025](https://doi.org/10.1039/D4MO00241E).

+ The source code, environment configuration file (dependencies), and documentation will be here soon! I am hoping to get that out early 2026 at the latest. After that, for transparency, the code written for the model development and training process will also be provided "as-is" as a separate repository and linked to from here with less guarantee for functionality/reusability.

+ The results in the pre-print make it clear that, in its current state, this tool is not competitive with current state-of-the-art *ab initio* gene detection tools. That said, a more polished version of this approach may have potential. I am mainly providing the source code to document my effort, or in case anyone finds this idea promising and wants to build on it. I also hope to keep improving this tool past the performance demonstrated in the pre-print/manuscript, e.g., via an HMM-based post-processing model coupled with the core physicochemical classification model. The post-processing algorithm is likely the weakest link of the current tool.

+ Please contact **jhgmbioinfo@gmail.com** with any questions about the pre-print or source code.
