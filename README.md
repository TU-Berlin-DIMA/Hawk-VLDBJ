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

**Publication**

Paper: [Generating Custom Code for Efficient Query Execution on Heterogeneous 
Processors](https://doi.org/10.1007/s00778-018-0512-y)

BibTeX citation:
```
@Article{Breß2018,
    title="Generating custom code for efficient query execution on heterogeneous processors",
    author="Bre{\ss}, Sebastian and K{\"o}cher, Bastian and Funke, Henning and Zeuch, Steffen and Rabl, Tilmann and Markl, Volker",
    journal="The VLDB Journal",
    year="2018", month="Jul", day="09",
    issn="0949-877X",
    doi="10.1007/s00778-018-0512-y",
    url="https://doi.org/10.1007/s00778-018-0512-y"
}
```

## How to build source code and setup reference databases
After cloning the repository you can find the source code in the directory `source`.

**System Requirements**
We recommend Ubuntu (14.04.3 LTS or higher, 64 bit) as operating system with g++ (version 4.8 or higher) compiler.
To install all required tools and libraries you can use the script 
[`utility_scripts/install_cogadb_dependencies.sh`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/blob/master/source/utility_scripts/install_cogadb_dependencies.sh).
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
You can use the script 
[`utility_scripts/setup_reference_databases.sh`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/blob/master/source/utility_scripts/setup_reference_databases.sh) 
to download and setup the reference databases for the experiments.
You will be ask where you want to store them. 


## How to run the experiments
All scripts for the experiments will ask for the pathes to the CoGaDB executable, to the TPC-H database and to the SSB database.
Files related to the exploration experiments are located in the folder 
[`Hawk-VLDBJ/benchmarks/exploration`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/exploration).
All files for the Falcon Query Experiment can be found in the folder 
[`Hawk-VLDBJ/benchmarks/falcon_query_experiments`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/falcon_query_experiments).

### Exploration Experiments

#### Full Exploration
In this experiment we execute all possible variants for queries defined in the subfolder [`full_exploration_queries`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/exploration/full_exploration_queries) as `*.coga` file.
As result of this experiment we get the execution time of each variant per query in the subfolder `results`.

#### Feature Wise Exploration
In multiple iterations we execute all possible values per dimension.
At the end we have found the variant that performs best over all executed queries.
We explain the algorithm in detail in the paper.
The queries are configured by a `*.coga` file in the subfolders 
[`ssb_queries`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/exploration/ssb_queries) and 
[`tpch_queries`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/exploration/tpch_queries).
You can perform the experiment with the script 
[`execute_feature_wise.sh`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/blob/master/source/benchmarks/exploration/execute_feature_wise.sh).
The best variant for each recognized device can be found in the subfolder `results`.

#### Feature Wise per Query Exploration
In this experiement we find the best performing variant per query.
Apart from that we use the same algorithm as for the feature-wise experiment.
You can perform the experiment with the script 
[`execute_feature_wise_per_query.sh`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/blob/master/source/benchmarks/exploration/execute_feature_wise_per_query.sh).
We search the best performing variants for the queries configured in directories
[`ssb_queries_feature_wise_per_query`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/exploration/ssb_queries_feature_wise_per_query) and 
[`tpch_queries_feature_wise_per_query`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/exploration/tpch_queries_feature_wise_per_query).
The best variant for each query and recognized device can be found in the subfolder `results`.

### Falcon Query Experiments
All queries that are configured by a `*.coga` file in the subfolder 
[`ssb_queries`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/falcon_query_experiments/ssb_queries) and 
[`tpch_queries`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/falcon_query_experiments/tpch_queries) are executed.
Therefore the variants in the subfolder [`variants`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/tree/master/source/benchmarks/falcon_query_experiments/variants) are 
used.
You can perform the experiment with the script 
[`run_experiments.sh`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/blob/master/source/benchmarks/falcon_query_experiments/run_experiments.sh).

#### What is done by the script?
- Available devices are detected.
- Variants are used to run all available SSB and TPC-H queries on detected devices.

#### Analysing the Results
- Change to the subfolder with experimental results `falcon_paper_experimental_results-*`
- Execute scripts to collect results: 
[`../collect_results.sh`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/blob/master/source/benchmarks/falcon_query_experiments/collect_results.sh) 
and 
[`../collect_results_compilation_time.sh`](https://github.com/TU-Berlin-DIMA/Hawk-VLDBJ/blob/master/source/benchmarks/falcon_query_experiments/collect_results_compilation_time.sh).
