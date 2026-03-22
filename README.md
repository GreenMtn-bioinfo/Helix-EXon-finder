# Helix-EXon-finder

**H**elix-**EX**on-finder, or **HEX**-finder, is a command-line tool that uses a deep learning model trained on physicochemical/structural profiles estimated from DNA sequences for *ab initio* detection of exons in genomic sequences. It is only a proof-of-concept attempt, which is provided here so that others can easily try it out.

+ The models and algorithms at the heart of this tool were developed, trained, and evaluated for my capstone project in the Brandeis GPS Bioinformatics program, during an 8-week independent research course (RBIF 120: Research Topics in Computational Biology). Please see the [preprint](https://doi.org/10.64898/2025.12.19.694709) written about that work for more details.

    + My research project was inspired by and based on published work by [Mishra et al. in 2021](https://doi.org/10.1093/nar/gkab098), as well as follow-up published work by [Sharma et al. in 2025](https://doi.org/10.1039/D4MO00241E). I do **not** claim to be the original author or owner of the underlying approach, or any external supplemental data used by this repository. I am a curious independent evaluator of an idea that seems interesting, and so I made my own implementation of it to learn more. My project was only possible in a timely manner due to the fact that the authors publicly shared their tri- and tetra-nucleotide mappings to structural parameters (see [Data Dependencies](#data-dependencies) for more details). Likewise, evaluation of their model/pipeline during my project was also only possible because of their GitHub repository: [ChemEXIN](https://github.com/rnsharma478/ChemEXIN).

+ A separate repository holds the code written to train and evaluate the models at the heart of HEX-finder, which was the bulk of the work detailed in the [preprint](https://doi.org/10.64898/2025.12.19.694709). Please see [RBIF120-HEX-finder-training](https://github.com/GreenMtn-bioinfo/RBIF120-HEX-finder-training) and its corresponding documentation for more details.

+ The results in the [preprint](https://doi.org/10.64898/2025.12.19.694709) make it clear that, in its current state, HEX-finder and its underlying models are not competitive with state-of-the-art *ab initio* gene detection tools. This is unsurprising, given that these models were trained on data that only encapsulates one of many aspects of a gene's layout, and they only make nucleotide-level predictions that need to be interpreted/processed further. That said, incorporation of this type of physicochemical/structural information into a more complex and refined approach may have potential. I am mainly providing the source code to document my effort, or in case anyone finds this implementation interesting and wants to evaluate or build on it.

+ I hope to keep improving this tool to see if it can achieve better performance than what was demonstrated in the preprint. It is clear that the models at the heart of this tool cannot reach their full potential without a robust and biologically-motivated post-processing algorithm to sort the signal from the noise in the nucleotide-level predictions. For example, an HMM-based post-processing model coupled with one of the existing structural classification models could result in an improved performance, e.g. by identifying the sets of exon predictions that are more probable once exon length distributions, reading frame consistency, biological viability, etc. are considered.

+ Please contact **jhgmbioinfo@gmail.com** with any questions about the [preprint](https://doi.org/10.64898/2025.12.19.694709) or this repository.

# Quick Start Guide

## Pre-requisites

Before attempting to set up HEX-finder, you need the following:

1) An **NVIDIA GPU** with an **official proprietary NVIDIA driver** already properly installed on your system. While not strictly necessary, HEX-finder was developed and tested on systems with NVIDIA GPUs, which TensorFlow/Keras utilizes via the CUDA Toolkit and NVIDIA cuDNN. Tensorflow can and will use your CPU if no GPU is detected, but this mode of operation is not recommended. You do not need the newest or fanciest GPU, but any GPU will make the inference step *at least* an order of magnitude faster than with a CPU, and inference is a rate-dominating step of this pipeline. Please see [Hardware and Runtimes](#hardware-and-runtimes) for a few reference points on what to expect regarding run duration when using a GPU. If running "`nvidia-smi`" in your terminal shows a recent driver version and an accurate list of your GPU(s), you are probably good to go!
    + **For AMD GPUs**: It may be entirely possible to use this tool with an AMD GPU, but it is untested at the moment. I have provided an alternative environment file called [environment-amd.yml](environment-amd.yml), which should provide a sound starting point analogous to the NVIDIA-compatible environment setup in [environment.yml](environment.yml). That alternative file may work out of the box, but will likely require a slight modification to the version numbers on one or two lines to work with your specific hardware. Please see the official AMD documentation on [TensorFlow + ROCm installation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/tensorflow-install.html) for more details on which versions of ROCm (and which hardware) are compatible with specific TensorFlow versions. HEX-finder was only developed and tested with TensorFlow 2.19, so bear that in mind when adjusting version numbers in [environment-amd.yml](environment-amd.yml) to work with your hardware. That said, there is likely some flexibility in the TensorFlow versions that will work. Just as for NVIDIA users, the setup instructions below still expect you to already have a properly installed GPU driver on your Linux system. Unlike for NVIDIA GPUs, you will need an open-source amdgpu kernel driver for this setup to work, rather than a legacy proprietary driver.

2) **A Linux or Unix-like system with Bash**
    + This could be inside WSL/WSL2 or a virtual machine.
    + HEX-finder has only been tested in WSL (running Ubuntu 22.04.2 LTS) and several native Debian-based systems.

3) **Miniconda or Conda**
    + Please see the official [quickstart Miniconda installation instructions for Linux](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-2).

4) **Git**
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
+ If you have an AMD GPU, you will first need to review and possibly edit [environment-amd.yml](environment-amd.yml) to match your hardware requirements before proceeding (see the "**For AMD GPUs**" note above under [Pre-requisites](#pre-requisites) for more context). After that is done, run "`bash setup.sh amd`" or "`bash setup.sh AMD`" instead of the command below.

```bash
bash setup.sh # This assumes NVIDIA hardware
```

Activate the conda environment that was just created. This environment will always need to be active when using HEX-finder:

```bash
conda activate HEX-finder
```

You are now ready to use HEX-finder! There are two ways to proceed:

1) **Start using the HEX-finder or its utilities with no further setup commands.** If you choose this route, you will need to run HEX-finder via commands like "`python HEX-finder.py predict -f /path/to/a/FASTA.fasta`" or "`./HEX-finder.py predict -f /path/to/a/FASTA.fasta`". Those examples assume you are running them from within the repo's root directory.

    **OR**

2) **Conduct a pip editable install of HEX-finder:**

    ```bash
    pip install -e . # Still from within the repo's root directory!
    ```
    This second option is mainly for convenience. It lets you instead run commands like `HEX-finder predict -f /path/to/a/FASTA.fasta` from anywhere on your system with full tab completion capabilities, and without having to specify the full path to the repo/script.

No matter which option you choose, all outputs, such as structural profiles, predictions, log files, etc., will be stored in their designated directories of the repository. This tool is designed with flexibility for the user to specify input file locations, but not the output file locations. If for some reason you intend to repeatedly run HEX-finder, e.g. for many FASTA files or with different run options, make sure to add copy commands to whatever pipeline is calling HEX-finder in order to move output files elsewhere, prior to running the tool again (which automatically clears the output directories).

Please see the next section for more details about using HEX-finder and its auxiliary tools.


## Using HEX-finder

There are three utilities/sub-commands accessible under the HEX-finder command:

1) **`HEX-finder fetch`:** Use this if you do not already have sequences on hand in a properly-formatted FASTA file and you would like to quickly retrieve some from GrCh38.p14 to try out `HEX-finder predict`. This sub-command can be used to randomly sample a number of sequences from within the regions of the genome that were excluded from the model's training data. This mode requires a file of allowed sequence lengths (see [example_length_distribution](/demo_sequences/example_length_distribution) for a valid example). Alternatively, you can pass this utility a file of valid coordinates, and it will retrieve the sequences for the specified regions and strands of the genome (see [Figure11_demo_coords](/demo_sequences/Figure11_demo_coords) for a valid example). In this mode, you will be warned if any coordinates overlap with a training zone (see the [preprint](https://doi.org/10.64898/2025.12.19.694709)). No matter how sequences are retrieved, this tool maps global coordinates of RefSeq MANE Select exons to local, 1-based coordinates for each sequence and saves them all to one GFF. This file can be used later as a source of truth with `HEX-finder visualize` to qualitatively benchmark the performance of `HEX-finder predict` against the relevant reference features. With you permission, this utility can also automatically fetch and prepare a copy of GRCh38.p14 and the corresponding annotation from NCBI if you do not already have these files. If you use this tool to get your sequences, for convenience their IDs will be the same as their genomic coordinates, but neither `HEX-finder predict` or `HEX-finder visualize` require this.

2) **`HEX-finder predict`:** This calls the core HEX-finder pipeline to make exon-level predictions for all processable sequences in a FASTA file. This is functionally the same pipeline developed and evaluated in the [preprint](https://doi.org/10.64898/2025.12.19.694709). One GFF file per sequence is generated and saved in the [predictions/](/predictions/) directory, as well as a log of any sequences skipped (and why). Prediction coordinates are local (i.e. the number of nucleotides from the sequence start) and 1-based. Structural profiles for each sequence are saved as NPY files in [profiles/](/profiles/). If desired, these can be automatically deleted once `HEX-finder predict` is finished. This tool is not necessarily dependent on the output of the other two utilities. The others simply facilitate fetching test sequences and visualizing the predictions for them against known features in GrCh38.p14.

3) **`HEX-finder visualize`:** Makes an HTML report with an embedded Plotly graph for easily viewing the predictions from `HEX-finder predict` (i.e. the GFFs sitting in the [predictions/](/predictions/) directory). Optionally, you can specify a path to a GFF file with local, 1-based coordinates of known true features for direct comparison to HEX-finder's predictions. `HEX-finder fetch` will prepare this truth/reference GFF file automatically if you use it to fetch your sequences. Otherwise, you need to handle the conversion to local and 1-based coordinates, as well as the mapping of features to sequence IDs, in order to make a comparable GFF file yourself. In that case, I strongly recommend checking out the example file that ships with this repository ([Figure11_demo_coords_truth_features_local.gff](/demo_sequences/Figure11_demo_coords_truth_features_local.gff)) and comparing it to the corresponding FASTA file to make sure your truth GFF will work with this utility. Sequence IDs in the truth/reference GFF do *not* need to be genomic coordinates, `HEX-finder fetch` just does this for traceability and convenience. They only need to match the sequence IDs in your corresponding FASTA.
    + This tool was provided to be transparent about HEX-finder's performance, which is currently far from optimal. A genome browser like [IGV](https://igv.org/) could be used to do the same visual evaluation, and IGV was used throughout the project. That said, visualizing predictions against truth features can get a bit confusing for features that happen to fall on the reverse strand of the reference genome. This tool naturally displays all sequences and features in a way that is agnostic to their source genome strand convention. It takes a little more work on the truth feature preparation end, but I think this is worth the visual clarity/consistency.
    + The report has an easy way to copy the sequence IDs for the currently plot, which Plotly does not allow by default. If the sequences were retrieved by `HEX-finder fetch`, the ID will be a genomic coordinate that can be copy-pasted (minus the strand symbol) into IGV's coordinate window for quick vetting of that plot, assuming you have GRCh38.p14 and RefSeq features loaded in IGV. When doing this, bear in mind that if the sequence came from the reverse strand, IGV will display features in reverse order compared to the Plotly graph!

To get familiarized with these sub-commands, you can follow along with the two examples below. For these examples, a pip editable install will be assumed. If you skipped that, all of these commands should work by simply changing "`HEX-finder`" to "`python HEX-finder.py`". Also, it will be assumed these commands are being run from the repo's root directory to keep paths short and generic, even though this is not necessary.

### Example A

**Recreate Figure 11 from the preprint**

1) Fetch the desired sequences from the genome using the properly-formatted [coordinates file](/demo_sequences/Figure11_demo_coords) already provided:

    ```bash
    HEX-finder fetch --coordinates ./demo_sequences/Figure11_demo_coords
    ```

    + If this is your first time running using `fetch`, it will ask if you want to download the genome from NCBI and prepare it. Choose "yes" unless you already known you have the correct files, in which case retry with the `-r/--reference_dir` option. Automatically downloaded files will be kept and reused in the future, so that is a one-time step if you choose it. If you already have your own reference files elsewhere and you do not want to repeatedly specify `-r/--reference_dir`, you could place symbolic links in [data/reference_genome/](/data/reference_genome/) to your copies of the genome, index, and annotation.
    + This command creates the files [Figure11_demo_coords.fasta](/demo_sequences/Figure11_demo_coords.fasta) and [Figure11_demo_coords_truth_features_local.gff](/demo_sequences/Figure11_demo_coords_truth_features_local.gff) in the [demo_sequences/](/demo_sequences/) directory.

2) Make predictions for the sequences in the newly prepared FASTA file:

    ```bash
    HEX-finder predict --fasta ./demo_sequences/Figure11_demo_coords.fasta -m MBDA-Net -t 0.7
    ```
    + Here, the model is changed from the default (using `-m/--model`), as is the exon-level acceptance threshold for predictions (using `-t/--threshold`). The default model (TCN) is recommended for speed, especially on older GPUs, but this is how the predictions were made for Figure 11 in the [preprint](https://doi.org/10.64898/2025.12.19.694709).

3) Visualize the predictions and compare them to RefSeq exons, which were automatically retrieved and prepared during step 1:

    ```bash
    HEX-finder visualize --truth_features ./demo_sequences/Figure11_demo_coords_truth_features_local.gff -tl gene
    ```
    + After running that command, an HTML file can now be found in [predictions/](/predictions/), unless you used the `-o/--output_path` option to write it elsewhere. Open that file in a web browser, use the dropdown menu, and double-click on a sequence ID to load that sequence's features to the plot. You will see that the plots on that page are consistent with the ones seen in Figure 11 (aside from the absence of ChemEXIN predictions and some style changes).
    + The `-tl/--truth_labels_attribute` argument tells the utility which variable in the GFF's attribute column to use for labeling individual RefSeq exons. In this case, passing "gene" results in reference exons labeled with their HUGO gene symbol on each graph, but any valid attribute in that column could be used. The convention of ";" for attribute separation and "=" for attribute assignment is expected in the GFF file when using this option.

### Example B

**Make predictions for 20 randomly-sampled sequences**

1) This time we will use different arguments to sample 20 sequences of random length from the held-out regions of GrCh38.p14, using [example_length_distribution](/demo_sequences/example_length_distribution) for the allowed lengths. If you have a system with no dedicated GPU, which is not recommended for running HEX-finder, then I strongly recommend making a copy of [example_length_distribution](/demo_sequences/example_length_distribution) with shorter short sequence lengths (≤ 500 nucleotides) and following this example using that file instead (see [Pre-requisites](#pre-requisites) and [Hardware and Runtimes](#hardware-and-runtimes)):

    ```bash
    HEX-finder fetch --number 20 --lengths ./demo_sequences/example_length_distribution --seed 1010 --output 20_seqs_seed_1010
    ```
    + The `-o/--output` argument specified above only specifies *prefix* for the names of the three output files created in this step, which are automatically saved in [demo_sequences/](/demo_sequences/).
    + This command creates the files "20_seqs_seed_1010_coords", "20_seqs_seed_1010.fasta", and "20_seqs_seed_1010_truth_features_local.gff" in the [demo_sequences/](/demo_sequences/) directory.

2) Now we can make predictions for the sequences in the new FASTA, using the default model (TCN) and exon-level score cutoff (0.75):

    ```bash
    HEX-finder predict -f ./demo_sequences/20_seqs_seed_1010.fasta -d
    ```
    + We used `-d/--delete` to delete the structural profiles (NPY files) in [profiles/](/profiles/) after inference. Using this option moves any log files from [profiles/](/profiles/) to [predictions/](/predictions/) before clearing the profiles directory. While not a problem in this example, the profiles can take up a lot of space when we process very many and/or very long sequences. For example, ~0.577 Gnc of sequence takes up ~90 GB of disk space. 
    + HEX-finder currently skips sequences with any number of N's or other characters outside of the main four for DNA (A, C, G, and T). Because of this, the wider you cast your "sampling net" when using `HEX-finder fetch` with GrCh38, the more likely you are to have fewer files in [profiles/](/profiles/) and [predictions/](/predictions/) than there were sequences in your input FASTA. Interpolation for short stretches of N's is already technically possible and may be implemented at a later date. Check [profiles/](/profiles/) for a log file of skipped sequences (or check [predictions/](/predictions/) if you used `-d/--delete`). 
        + Individual sequences with a length of <104 or >10 M nucleotides will also be skipped. The maximum is somewhat arbitrary and could easily be changed in the source code. The minimum is a hard limit based on the profile generation algorithm, the model's input window length, and the fact that this pipeline makes exon-level predictions. Making a prediction for a 104 nucleotide sequence is the bare minimum for the model to be able to call an exon start and end for two adjacent nucleotides around the center of the input sequence. **It is recommended to keep all input sequences approximately ≥150 nucleotides in length, even though the hard limit is lower.**

3) We can again visualize the predictions and compare them to the RefSeq exons retrieved and mapped to each sequence in step 1:

    ```bash
    HEX-finder visualize -t ./demo_sequences/20_seqs_seed_1010_truth_features_local.gff -tl gene -se -js -ac
    ```
    + Here we are using some more features not shown in the first example:
        + `-se/--skip_empty`: Prevents the script from generating any plot for sequences that have no HEX-finder predictions *and* no benchmark/truth features (if provided). In theory, this is useful for scenarios with many sequences that are from intergenic regions or long introns. Unfortunately `HEX-finder predict` currently produces enough false positives that empty plots are unlikely for long sequences, though it does still happen on occasion. This option becomes very useful when you are processing many shorter sequences, roughly on the order of ≤ 5000 nucleotides, as HEX-finder successfully identifying a sequence with no exons is more likely to occur (i.e. same average FPR, but multiplied by a lower number of input windows/"trials"). If any plots were skipped, the IDs of featureless sequences are logged in a file saved under [predictions/](/predictions/).
        + `-js/--javascript_included`: Tells Plotly to include its JavaScript in the HTML file, rather than a CND URL. This makes the report function properly without internet access, at the cost of a larger file size. 
        + `-ac/--accessibility_colors`: Changes the plot and legend colors to be more useful for people with various types of color blindness. I am no expert on accessibility or color palettes, but this option should provide a major improvement over the default for most. These plots already have a bit of dual-coding in their layout, which should also help.

    + If you followed along with the exact commands for this example up to this point, and looked through the report plots, you will have noticed that HEX-finder is far from reliable in its predictions. For long sequences, like the 1M nc sequence in the 9th plot (NC_000013.11:52757837-53757836(+)), HEX-finder spits out a bunch of scattered false positives, even though there are truly only a few exons in the entire sequence (you may need to zoom to see this). Meanwhile, for shorter ones, like the 60K nc sequence in the 13th plot (NC_000017.11:8133705-8193704(-)), it can accurately catch a few clusters of near-median length exons without excessive false positives. The model has technically not been shown either of these sequences during training, even though homology between sequences in the training and testing regions could complicate this assumption. Regardless, even a well-trained nucleotide-level classifier that performed strongly during testing would require additional biological rules and data to infer reliable exon- and gene-level predictions from the multiplicity of possible acceptor-donor combinations. As such, this tool, and the work it was based on, are really only a starting point on the road to seeing if the underlying structural approach has practical potential.

[Example A](#example-a) and [Example B](#example-b) should be enough to get you started with HEX-finder. Again, you do not need to use all three utilities. You could bring your own sequences in a FASTA, run `HEX-finder preidict` on that file, and then use `HEX-finder visualize` to look at the predictions alone. If you had your own GFF file of truth/benchmark features for those sequences in the correct format, you could use `HEX-finder visualize` to compare predictions to the features in that file. You could forgo `fetch` and `visualize` entirely and visualize, evaluate, or otherwise use the predictions in your own way. These utilities are provided together as an exercise in transparency and convenience.

Each of the sub-commands have a number of options not demonstrated in the examples above. The command-line help output is quite detailed, so for further practical information use `-h/--help` with any of the HEX-finder sub-commands in the terminal.


# Hardware and Runtimes

Currently, the exon prediction pipeline at HEX-finder's core is structured decently in terms of speed/efficiently. Expanding a batch of potentially very many and very long strings (i.e. DNA sequences) into 2D arrays of floating-point values (i.e. "structural profiles") and then performing inference on each array using a relatively small input window is an inherently intensive task, especially in Python. The structural profile generation step was implemented using multi-threading, as well as other tricks, so it benefits greatly from the number of cores typically available on modern CPUs. There are a few major improvements yet to be made to the post-processing algorithm that finalizes the exon predictions, but this is a relatively light-weight step, so inefficient implementation in Python is tolerable for now. **The rate-dominating step is model inference**, which is why a GPU is strongly recommended in order to get results in a reasonable amount of time.

Here are the specifications for two different systems and how long HEX-finder took to make predictions for the same sequence set on either system:

+ **Sequence Set:** 10,179,272 nucleotides of total sequence length spread over of 28 sequences, with individual lengths drawn from [example_length_distribution](/demo_sequences/example_length_distribution). The times below include inference and post-processing to get the final exon-level predictions (using the default TCN model with a threshold of 0.75):

    + **LAPTOP with NVIDIA GTX 1060:** 20.7 minutes
        + 12-thread CPU (2.2 GHz nominal, ~4 GHz boosted multi-core), 32 GB DDR4 RAM, NVIDIA GTX 1060 (6 GB GDDR5 VRAM)

    + **DESKTOP with NVIDIA RTX 3090:** 3.82 minutes
        + 12-thread CPU (2.5 GHz nominal, ~4 GHz boosted multi-core), 64 GB DDR5 RAM, NVIDIA RTX 3090 (24 GB GDDR6X VRAM)

Even for a decent consumer-grade CPU, in my experience, the prediction times are at least >100 minutes for the same sequence set.


# Data Dependencies

The `data/` directory included with this repo contains a number of key files required for HEX-finder to run. Some of these files include data originating from publicly available external sources and I do **not** take credit for, or claim ownership of, this data:

+ [**trinucleo_Sharma_et_al_2025_params.csv**](/data/trinucleo_Sharma_et_al_2025_params.csv) **&** [**tetranucleo_Sharma_et_al_2025_params.csv**](/data/tetranucleo_Sharma_et_al_2025_params.csv)**:** These files contain the exact same data as the physicochemical mappings calculated by [Sharma et al. in 2025](https://doi.org/10.1039/D4MO00241E), but split into two CSVs (by k-mer) for easier access and parsing by the profile generation algorithm at the heart of HEX-finder. The original supplemental data from the authors is publicly available at the publisher's website [here](https://www.rsc.org/suppdata/d4/mo/d4mo00241e/d4mo00241e1.xlsx). As stated above and in the [preprint](https://doi.org/10.64898/2025.12.19.694709), these data are the mappings that translate tri- and tetra-nucleotides into numerical values for each structural parameter, prior to the moving average calculation that leads to the final profiles. My capstone project would not have been possible without the authors' choice to be transparent and share this supplemental data publicly. I recommend checking out their tool, [ChemEXIN](https://github.com/rnsharma478/ChemEXIN), as well as the corresponding [publication](https://doi.org/10.1039/D4MO00241E).

+ [**human_exon_length_distribution_Mokry_et_al_2010.csv**](/data/human_exon_length_distribution_Mokry_et_al_2010.csv)**:** Contains data points estimated from Figure 1 in a publication by [Mokry et al. in 2010](https://doi.org/10.1093/nar/gkq072). This was done using a tool that facilitates estimating data points from images of 2D plots. This approach is functionally analogous to using a ruler to estimate the relationship between axis units and absolute length units for a graph on paper, and then estimating the coordinates for each point via ruler measurements and conversion. I do not claim that this data is a high-fidelity recreation of the original data portrayed in Figure 1 of that paper. This approach was a crude but quick stand-in for higher quality data on the distribution of exon lengths in the human genome at a point in the project where I was tight on time. I hope to replace it with a distribution that has been more rigorously calculated, e.g. using RefSeq MANE Select exon annotation. This will dovetail nicely into an improvement of the post-processing pipeline as a whole.

For more details on the development and training of the models behind HEX-finder, please see the [preprint](https://doi.org/10.64898/2025.12.19.694709) on that work.