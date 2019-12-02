# Generating Custom Code for Efficient Query Execution on Heterogeneous Processors
This repository contains our work on custom code generation for efficient query execution on heterogenous processors.
It was first published in the [VLDB Journal](http://vldb.org/vldb_journal).

**Abstract** Processor manufacturers build increasingly specialized processors to mitigate the effects of the power wall in order to deliver improved performance.
Currently, database engines have to be manually optimized for each processor which is a costly and error prone process.
In this paper, we propose concepts to adapt to and to exploit the performance enhancements of modern processors automatically.
Our core idea is to create processor-specific code variants and to learn a well-performing code variant for each processor.  
These code variants leverage various parallelization strategies and apply both generic and processor-specific code transformations.
Our experimental results show that the performance of code variants may diverge up to two orders of magnitude. 
In order to achieve peak performance, we generate custom code for each processor.
We show that our approach finds an efficient custom code variant for multi-core CPUs, GPUs, and MICs.

<!--
**Publications**
- Paper: [Generating Custom Code for Effi
cient Query Execution on Heterogeneous Processors](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/blob/master/Paper-Generating-Custom-Code-for-Effi
cient-Query-Execution-on-Heterogeneous-Processors.pdf)
- Poster: [Generating Custom Code for Effi
cient Query Execution on Heterogeneous Processors](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/blob/master/Poster-Generating-Custom-Code-for-Effi
cient-Query-Execution-on-Heterogeneous-Processors.pdf)

- BibTeX citation:
```
TO BE DONE
}
```
-->

## How to build source code and setup reference databases
After cloning the repository you can find the source code in the directory `source`.

**System Requirements**
We recommend Ubuntu (14.04.3 LTS or higher, 64 bit) as operating system with g++ (version 4.8 or higher) compiler.
To install all required tools and libraries you can use the script `utility_scripts/install_cogadb_dependencies.sh`.
All dependencies are installed via [APT package manager](https://wiki.ubuntuusers.de/APT/).
The prequisites in detail are:

- Tools
    - Flex
    - make
    - cmake (version 2.6 or higher)
    - Bison (supported from version 2.5)
    - clang and LLVM (supported from version 3.6)
    - Doxygen (documentation generation, optional)
    - Pdf Latex (documentation generation, optional)

- Libraries
    - OpenCL
    - Readline Library
    - Google Sparse Hash
    - Boost Libraries (version 1.48 or higher)
    - Boost Compute (automatically downloaded during make)
    - Google Test (automatically downloaded during make)
    - RapidJSON (automatically downloaded during make)
    - XSLT Library (documentation generation)

**Build Instructions**
We use cmake to generate a build system. Afterwards you can compile the code via the `make` command
 
```
user@host:~/Hawk-VLDBJ/source$ mkdir release_build && cd release_build
user@host:~/Hawk-VLDBJ/source/release_build$ cmake -DCMAKE_BUILD_TYPE=Release .. && make -j
```

**Setup Reference Databases**
To execute the experiments we need data.
We use the TPC-H and SSB reference databases as workload data.
You can use the script `utility_scripts/setup_reference_databases.sh` to download and setup the reference databases for the experiments.
You will be ask where you want to store them. 


## How to run the experiments
All scripts for the experiments will ask for the pathes to the CoGaDB executable, to the TPC-H database and to the SSB database.
Files related to the exploration experiments are located in the folder `Hawk-VLDBJ/benchmarks/exploration`.
Alls files for the Falcon Query Experiment can be found in the folder `Hawk-VLDBJ/benchmarks/falcon_query_experiments`.

### Exploration Experiments

#### Full Exploration

#### Feature Wise Exploration
In multiple iterations we execute all possible values per dimension.
At the end we have found the variant that performs best over all executed queries.
We explain the algorithm in detail in the paper.
The queries are configured by a `*.coga` file in the subfolders `ssb_queries` and `tpch_queries`.
You can perform the experiment with the script `execute_feature_wise.sh`.
The best variant for each recognized device can be found in the subfolder `results`.

#### Feature Wise per Query Exploration
In this experiement we find the best performing variant per query.
Apart from that we use the same algorithm as for the feature-wise experiment.
You can perform the experiment with the script `execute_feature_wise_per_query.sh`.
The best variant for each query and recognized device can be found in the subfolder `results`.

### Falcon Query Experiments
All queries that are configured by a `*.coga` file in the subfolder `ssb_queries` and `tpch_queries` are executed.
Therefore the variants in the subfolder `variants` are used.
You can perform the experiment with the script `run_experiments.sh`.

#### What is done by the script?
- Available devices are detected.
- Variants are used to run all available SSB and TPC-H queries on detected devices.

#### Analysing the Results
- Change to the subfolder with experimental results `falcon_paper_experimental_results-*`
- Execute scripts to collect results: `../collect_results.sh` and `../collect_results_compilation_time.sh`
