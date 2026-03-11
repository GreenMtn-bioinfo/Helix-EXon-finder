# Helix-EXon-finder

**H**elix-**EX**on-finder, or **HEX**-finder, is a command-line tool that uses a deep learning model trained on physicochemical/structural profiles estimated from DNA sequences for *ab initio* detection of exons in genomic sequences. It is only a proof-of-concept attempt, provided here so that others can easily evaluate it themselves on any sequence.

+ The models and algorithms at the heart of this tool were developed, trained, and evaluated for my capstone project in the Brandeis GPS Bioinformatics program, during an 8-week independent research course (RBIF 120: Research Topics in Computational Biology). Please see the [pre-print](https://doi.org/10.64898/2025.12.19.694709) written about that work for more details.

+ The independent research project leading to this repository was inspired by and based on published work by [Mishra et al. in 2021](https://doi.org/10.1093/nar/gkab098), as well as follow-up published work by [Sharma et al. in 2025](https://doi.org/10.1039/D4MO00241E). I do **not** claim to be the original author or owner of the underlying approach, or any supplemental data used by this repository. I am a curious independent evaluator of an idea that seems interesting/promising. My project was only possible in a timely manner due to the fact that the authors publicly shared their tri- and tetra-nucleotide mappings to structural parameters (see [Data Dependencies](#data-dependencies) for more details). Likewise, evaluation of their model/pipeline during my project was also only possible because of their GitHub repository: [ChemEXIN](https://github.com/rnsharma478/ChemEXIN).

+ The results in the [pre-print](https://doi.org/10.64898/2025.12.19.694709) make it clear that, in its current state, HEX-finder and its underlying models are not competitive with state-of-the-art *ab initio* gene detection tools. This is unsurprising, given that these models were trained on data that only encapsulates one of many aspects of a gene's layout. That said, incorporation of this type of physicochemical/structural information into a more complex and refined approach may have potential. I am mainly providing the source code to document my effort, or in case anyone finds this implementation interesting and wants to evaluate or build on it. I also hope to keep improving this tool past the performance demonstrated in the pre-print. For example, an HMM-based post-processing model coupled with one of the existing structural classification models could result in an improved performance, e.g. by identifying the *sets* of exon predictions that are more likely once exon length distributions, reading frame consistency, biological viability, etc. are considered.

+ A separate repository will be made to share the code written to train and evaluate the models at the heart of HEX-finder, which was the bulk of the work detailed in the [pre-print](https://doi.org/10.64898/2025.12.19.694709). That repository will come with no guarantee of portability/reproducibility, though basic documentation and a YAML to recreate the environment/dependencies will be provided. 

+ Please contact **jhgmbioinfo@gmail.com** with any questions about the [pre-print](https://doi.org/10.64898/2025.12.19.694709) or this source repository.


# Quick Start Guide

## Pre-requisites

Before attempting to set up this repository, you need the following:

1) An **NVIDIA GPU** with an **official proprietary NVIDIA driver** already properly installed on your system . While not strictly necessary, HEX-finder was developed and tested on systems with NVIDIA GPUs, which TensorFlow/Keras utilized via the CUDA Toolkit and NVIDIA cuDNN. Tensorflow can and will use your CPU if no GPU is detected, but this mode of operation is not recommended. You do not need the newest or fanciest GPU, but any GPU will make the inference step *at least* an order of magnitude faster than with a CPU, and inference is a rate-dominating step of this pipeline. Please see [Hardware and Runtimes](#hardware-and-runtimes) for a few reference points on what to expect regarding speed using a GPU. If running `nvidia-smi` in your terminal shows a recent driver version and an accurate list of your GPU(s), you are probably good to go!
    + **For AMD GPUs**: It may be entirely possible to use this tool with an AMD GPU, but it is untested at the moment. I have provided an alternative environment file called [environment-amd.yml](environment-amd.yml), which should provide a sound starting point analogous to the NVIDIA-compatible environment setup in [environment.yml](environment.yml). That alternative file may work out of the box, but will likely require a slight modification to the version numbers on one or two lines to work with your specific hardware. Please see the official AMD documentation on [TensorFlow + ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html) for more details on which versions of ROCm (and which hardware) are compatible with specific TensorFlow versions. HEX-finder was only developed and tested with TensorFlow 2.19, so bear that in mind when adjusting version numbers in [environment-amd.yml](environment-amd.yml) to work with your hardware. That said, there is likely some flexibility in the TensorFlow versions that will work. Just as for NVIDIA users, the setup instructions below still expect you to already have a properly installed GPU driver on your Linux system. That said, unlike for NVIDIA, you will need an open-source amdgpu kernel driver for this setup to work, rather than a legacy proprietary driver.

2) A Linux or Unix-like system with Bash
    + This could be inside WSL/WSL2 or a virtual machine.
    + HEX-finder has only been tested in WSL (running Ubuntu 22.04.2 LTS) and several native Debian-based systems.

3) Miniconda or Conda
    + Please see the official [basic](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer) or [quickstart](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2) Miniconda installation instructions for Linux.

4) Git
    + If you do not already have it available in your command-line, please see the [Git](https://git-scm.com/install/linux) or [GitHub](https://github.com/git-guides/install-git) instructions for your specific Linux or Unix-like system.


## Installation and Setup Instructions

Make the directory where you would like the repo to live and cd into it. For example:

```bash
mkdir ~/My_Git_Clones/
cd ~/My_Git_Clones/
```

Clone this repository using the following command in your terminal and then cd into it:

```bash
git clone https://github.com/GreenMtn-bioinfo/Helix-EXon-finder
cd ./Helix-EXon-finder/
```

Run the setup script, which will automatically build and configure the conda environment (assuming NVIDIA hardware):
+ If you have an AMD GPU, you will first need to review and possibly edit [environment-amd.yml](environment-amd.yml) to match your hardware requirements before proceeding (see the "**For AMD GPUs**" note above under [Pre-requisites](#pre-requisites) for more context). After that is done, run `bash setup.sh amd` or `bash setup.sh AMD` instead of the command below.

```bash
bash setup.sh # This assumes NVIDIA hardware
```

Activate the conda environment that was just created. This environment will always need to be active when using HEX-finder:

```bash
conda activate HEX-finder
```

You are now ready to use HEX-finder! There are two ways to proceed:

1) Start using the HEX-finder or its utilities with no further setup commands. If you choose this route, you will need to run HEX-finder via commands like `python HEX-finder.py predict -f /path/to/a/FASTA.fasta` or `./HEX-finder.py predict -f /path/to/a/FASTA.fasta`. These examples assume you are running them from within the repo's root directory.

    **OR**

2) Conduct a pip editable install of HEX-finder:

    ```bash
    # Still from within the repo's root directory!
    pip install -e .
    ```
    This second option is mainly just for convenience. It lets you instead run commands like `HEX-finder predict -f /path/to/a/FASTA.fasta` from anywhere on your system with full tab completion capabilities, and without having to specify the full path to the repo/script.

No matter which option you choose, all outputs, such as structural profiles, predictions, log files, etc., will be stored in their designated directories of the repository. For now, this tool designed with much flexibility for the input file locations, but not for the output file locations. If for some reason you intend to repeatedly run HEX-finder, e.g. for many FASTA files or with different run options, make sure to add copy commands to whatever pipeline is calling HEX-finder in order to move output files elsewhere, prior to running the tool again (which automatically clears the output directories).

Please see [Usage Examples](#usage-examples) for more details about using HEX-finder and its auxiliary tools.


# Usage Examples

A detailed introduction on how to use HEX-finder and its utilities is coming soon!

For now, the command-line help output is quite detailed, so for more information run `python HEX-finder.py -h` or `HEX-finder -h`, depending on your setup choices. You can also use -h/--help with any of the HEX-finder subcommands, and that is where most of the useful information is.


# Hardware and Runtimes

Currently, the exon prediction pipeline at HEX-finder's core is structured decently in terms of speed/efficiently. Expanding a batch of potentially very many and very long strings (i.e. DNA sequences) into 2D arrays of floating-point values (i.e. "structural profiles") and then performing inference on each array using a relatively small input window is an inherently intensive task. The structural profile generation step was implemented using multi-threading, as well as other tricks, so it benefits greatly from the number of cores typically available on modern CPUs. There are a few major improvements yet to be made to the post-processing algorithm that finalizes exon predictions, but this is a relatively light-weight step, so inefficient implementation in Python is tolerable. The rate-dominating step is model inference, which is why a GPU is strongly recommended in order to get results in a vaguely reasonable time.

Here are the specs for two different systems and how long HEX-finder took to make predictions for the same sequence set on either system:

+ **Sequence Set:** 10,179,272 bp of total sequence length spread over of 28 sequences, with individual lengths drawn from [example_length_distribution](/demo_sequences/example_length_distribution). Times include inference and post-processing to get the final exon-level predictions:

    + **LAPTOP w/ NVIDIA GTX 1060:** 20.7 minutes
        + 12-thread CPU (2.2 GHz nominal, ~4 GHz boosted multi-core), 32 GB DDR4 RAM, NVIDIA GTX 1060 (6 GB GDDR5 VRAM)

    + **DESKTOP w/ NVIDIA RTX 3090:** 3.82 minutes
        + 12-thread CPU (2.5 GHz nominal, ~4 GHz boosted multi-core), 64 GB DDR5 RAM, NVIDIA RTX 3090 (24 GB GDDR6X VRAM)

Even for a decent consumer-grade CPU, in my experience, the prediction times are at least >100 minutes for the same sequence set.


# Data Dependencies

The `data/` directory in this repo contains a number of key files required for HEX-finder to run. Some of these files include data originating from publicly available external sources and I do **not** take credit for, or claim ownership of, this data:

+ [trinucleo_Sharma_et_al_2025_params.csv](/data/trinucleo_Sharma_et_al_2025_params.csv) & [tetranucleo_Sharma_et_al_2025_params.csv](/data/tetranucleo_Sharma_et_al_2025_params.csv): These files contain the exact same data as the physicochemical mappings calculated by [Sharma et al. in 2025](https://doi.org/10.1039/D4MO00241E), but split into two CSVs (by k-mer) for easier access and parsing by the profile generation algorithm at the heart of HEX-finder. The original supplemental data from the authors is publicly available at the publisher's website [here](https://www.rsc.org/suppdata/d4/mo/d4mo00241e/d4mo00241e1.xlsx). As stated above, my capstone project would not have been possible without the authors' choice to be transparent and share this supplemental data publicly. I recommend checking out their tool, [ChemEXIN](https://github.com/rnsharma478/ChemEXIN), as well as the corresponding [publication](https://doi.org/10.1039/D4MO00241E).

+ [human_exon_length_distribution_Mokry_et_al_2010.csv](/data/human_exon_length_distribution_Mokry_et_al_2010.csv): Contains data points estimated from Figure 1 in a publication by [Mokry et al. in 2010](https://doi.org/10.1093/nar/gkq072). This was done using a tool that facilitates estimating data points from images of 2D plots. This approach was functionally analogous to using a ruler to estimate the relationship between axis units and absolute length units for a graph on paper, and then estimating the coordinates for each point via ruler measurements and conversion. I do not claim that this data is a high-fidelity recreation of the original data portrayed in Figure 1 of that paper. This approach was a crude but quick stand-in for higher quality data on the distribution of exon lengths in the human genome at a point in the project where I was tight on time. I hope to replace it soon with a distribution more rigorously calculated, e.g. using RefSeq exon annotation.

For more details on the development and training of the models behind HEX-finder, please see the [pre-print](https://doi.org/10.64898/2025.12.19.694709) on that work.