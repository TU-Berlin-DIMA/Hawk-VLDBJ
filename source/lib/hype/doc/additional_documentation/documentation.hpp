/***********************************************************************************************************
Copyright (c) 2012, Sebastian\htmlonly Bre&szlig;\endhtmlonly\latexonly
Bre{\ss}\endlatexonly, Otto-von-Guericke University of Magdeburg, Germany. All
rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
 **********************************************************************************************************/

/*!
 *  \file documentation.hpp
 *  \brief This file contains additional documentation, like the generated web
 * pages in the doxygen documentation.
 *  \author    Sebastian\htmlonly Bre&szlig;\endhtmlonly\latexonly
 * Bre{\ss}\endlatexonly
 *  \version   0.1
 *  \date      2012
 *  \copyright GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
 * http://www.gnu.org/licenses/lgpl-3.0.txt
 */

/*! \mainpage Documentation
 * \tableofcontents
 * \section intro_sec Introduction
 *
 * HyPE is a library build for automatic selection of processing units for
co-processing in database systems. The long-term goal of the project is to
implement a fully fledged query processing engine, which is able to
automatically generate and optimize a hybrid CPU/GPU physical query plan from a
logical query plan.
 *	It is a research prototype developed by the <a
href="http://www.uni-magdeburg.de/unimagdeburg/en/Otto_von_Guericke+University+Magdeburg.html">Otto-von-Guericke
University Magdeburg</a>
 * in collaboration with <a
href="http://www.tu-ilmenau.de/en/international/">Ilmenau University of
Technology</a>.
 *
 * See the publications listed below for more details.
 *
 * \section features_sec Features
 *
 * Currently, HyPE supports the following features:
 * - Entirely written in C++
 * - Decides on the (likely) optimal algorithm w.r.t. to a user specified
optimization criterion for an operation
 * - Unrestricted use in parallel applications due to thread-safety
 * - Easily extensible by utilizing a plug-in architecture
 * - <b>New:</b> Runs under Linux and Windows
 * - <b>New:</b> Supports the following compilers: g++ (>=4.5), clang, and
Visual C++
 * - Requires (almost) no knowledge about executed algorithms, just the relevant
features of the datasets for an algorithms execution time
 * - Collects statistical information to help the user to fine tune HyPE's
parameters for their use case
 * - Provides a parallel execution engine for operators

 *
 * 	\section sec_detailed_documentation Detailed Documentation
 *
 *    -# \subpage page_install_hype
 *    -# \subpage page_tutorial
 *    -# \subpage page_existing_plugins_hype
 *    -# \subpage page_configure_hype
 *    -# \subpage page_extending_hype
 *    -# \subpage page_faq
 *

 *
 * \section members_sec Project Members
 *
 * \subsection members_subsec Project Members:
 * - <a href="http://wwwiti.cs.uni-magdeburg.de/~bress/">Sebastian\htmlonly
Bre&szlig;\endhtmlonly\latexonly Bre{\ss}\endlatexonly</a> (University of
Magdeburg)
 * - Klaus Baumann  (University of Magdeburg)
 * - Robin Haberkorn (University of Magdeburg)
 * - Steven Ladewig (University of Magdeburg)
 * - Tobias Lauer (Jedox AG)
 * - <a href="http://wwwiti.cs.uni-magdeburg.de/~saake/">Gunter Saake</a>
(University of Magdeburg)
 * - <a
href="http://www.infosun.fim.uni-passau.de/spl/people-nsiegmund.php">Norbert
Siegmund</a> (University of Passau)
 * \subsection partners_subsec Project Partners:
 * - Felix Beier (Ilmenau University of Technology)
 * - Ladjel Bellatreche (LIAS/ISEA-ENSMA, Futuroscope, France)
 * - Hannes Rauhe (Ilmenau University of Technology)
 * - Kai-Uwe Sattler (Ilmenau University of Technology)
 * \subsection former_members_subsec Former Project Members:
 * - Ingolf Geist (University of Magdeburg)
 *
 *
 * \section publication_sec Publications
\htmlonly
<ul class="bib2xhtml">

<li><a name="thesisBress"></a>Sebastian Bre&szlig;.
<a
href="http://wwwiti.cs.uni-magdeburg.de/iti_db/publikationen/ps/auto/thesisBress.pdf">Ein
selbstlernendes Entscheidungsmodell f&uuml;r die Verteilung von
  Datenbankoperationen auf CPU/GPU-Systemen</a>.
Master thesis, University of Magdeburg, Germany, March 2012.
In German.</li>

<!-- Authors: Sebastian Bress and Siba Mohammad and Eike Schallehn -->
<li><a name="bress:2012:GvD:decision_model"></a>Sebastian Bre&szlig;, Siba
Mohammad, and
  Eike Schallehn.
<a
href="http://wwwiti.cs.uni-magdeburg.de/iti_db/publikationen/ps/auto/bress:2012:GvDB:decision_model.pdf">Self-Tuning
Distribution of DB-Operations on
  Hybrid CPU/GPU Platforms</a>.
In <i>Proceedings of the 24st Workshop Grundlagen von Datenbanken
  (GvD)</i>, pages 89&ndash;94. CEUR-WS, 2012.</li>

<!-- Authors: Siba Mohammad and Sebastian Bress and Eike Schallehn -->
<!-- <li><a name="siba:2012:GvD:cloud_dbms"></a>Siba Mohammad, Sebastian
Bre&szlig;, and
  Eike Schallehn.
<a href="http://ceur-ws.org/Vol-850/paper_mohammad.pdf">Cloud Data Management:
  a Short Overview and Comparison of Current Approaches</a>.
In <i>Proceedings of the 24st Workshop Grundlagen von Datenbanken
  (GvD)</i>, pages 41&ndash;46. CEUR-WS, 2012.</li>-->

<!-- Authors: Sebastian Bress and Eike Schallehn and Ingolf Geist -->
<li><a name="bress:towards_hybrid_query_processing"></a>Sebastian Bre&szlig;,
Eike Schallehn, and
  Ingolf Geist.
<a href="http://link.springer.com/chapter/10.1007/978-3-642-32518-2_3">Towards
  Optimization of Hybrid CPU/GPU Query Plans in Database Systems</a>.
In <i>Second ADBIS workshop on GPUs In Databases (GID)</i>, pages 27&ndash;35.
  Springer, 2012.</li>

<!-- Authors: Sebastian Bress and Felix Beier and Hannes Rauhe and Eike
  Schallehn and Kai Uwe Sattler and Gunter Saake -->
<li><a name="bress:adbis:2012"></a>Sebastian Bre&szlig;, Felix Beier, Hannes
  Rauhe, Eike Schallehn, Kai-Uwe Sattler, and Gunter Saake.
<a href="http://link.springer.com/chapter/10.1007/978-3-642-33074-2_5">Automatic
Selection of Processing Units for Coprocessing in Databases</a>.
In <i>16th East-European Conference on Advances in Databases and Information
  Systems (ADBIS)</i>, pages 57&ndash;70. Springer, 2012.</li>

<!-- Authors: Sebastian Bress and Ingolf Geist and Eike Schallehn and Maik Mory
  and Gunter Saake -->
<li><a name="BressGS+13"></a>Sebastian Bre&szlig;, Ingolf Geist, Eike
  Schallehn, Maik Mory, and Gunter Saake.
<a
href="http://control.ibspan.waw.pl:3000/contents/export?filename=Bress-et-al.pdf">A
Framework for Cost based Optimization of Hybrid CPU/GPU Query Plans in
  Database Systems</a>.
<cite>Control and Cybernetics</cite>, 41(4):715&ndash;742,
  2012.</li>

<!-- Authors: Sebastian Bress and Felix Beier and Hannes Rauhe and Kai Uwe
  Sattler and Eike Schallehn and Gunter Saake -->
<li><a name="BressBR+13"></a>Sebastian Bre&szlig;, Felix Beier, Hannes
  Rauhe, Kai-Uwe Sattler, Eike Schallehn, and Gunter Saake.
<a href="http://authors.elsevier.com/sd/article/S0306437913000732">Efficient
  Co-Processor Utilization in Database Query Processing</a>.
<cite>Information Systems</cite>, 38(8):1084&ndash;1096, 2013.
http://dx.doi.org/10.1016/j.is.2013.05.004.</li>

<!-- Authors: Sebastian Bress -->
<li><a name="Bress13"></a>Sebastian Bre&szlig;.
<a href="http://db.disi.unitn.eu/pages/VLDBProgram/pdf/PhD/p10.pdf">Why it is
  Time for a HyPE: A Hybrid Query Processing Engine for Efficient GPU
  Coprocessing in DBMS</a>.
In <cite>The VLDB PhD workshop</cite>. VLDB Endowment,
  2013.</li>

<!-- Authors: Sebastian Bress and Norbert Siegmund and Ladjel Bellatreche and
  Gunter Saake -->
<li><a name="BressSB+13"></a>Sebastian Bre&szlig;, Norbert Siegmund,
  Ladjel Bellatreche, and Gunter Saake.
<a href="http://link.springer.com/chapter/10.1007/978-3-642-40683-6_22">An
  Operator-Stream-based Scheduling Engine for Effective GPU Coprocessing</a>.
In <cite>17th East-European Conference on Advances in Databases and Information
  Systems (ADBIS)</cite>, pages 288&ndash;301. Springer,
  2013.</li>

<!-- Authors: Sebastian Bress and Max Heimel and Norbert Siegmund and Ladjel
  Bellatreche and Gunter Saake -->
<li><a name="BressHS+13"></a>Sebastian Bre&szlig;, Max Heimel, Norbert
  Siegmund, Ladjel Bellatreche, and Gunter Saake.
<a
href="http://link.springer.com/chapter/10.1007/978-3-319-01863-8_25">Exploring
the Design Space of a GPU-aware Database Architecture</a>.
In <cite>ADBIS workshop on GPUs In Databases (GID)</cite>, pages 225&ndash;234.
  Springer, 2013.</li>
</ul>
\endhtmlonly
\latexonly
\nocite{*}
\bibliographystyle{abbrv}
\bibliography{literature}
\endlatexonly
 *
*/

/*! \page page_install_hype Installation
 * \tableofcontents
 *
 * HyPE uses <a href="http://www.cmake.org/">CMake</a> as a cross-platform build
 *system.
 * Generally, building HyPE is similar on all platforms but we will nevertheless
 * highlight some platform-specifics on this page.
 *
 * There might also be precompiled binaries for your platform and
 *toolchain.<br/>
 * <b>Note</b> that C++ libraries for one platform (e.g. Windows) built with
 *different
 * toolchains or merely different toolchain versions are generally not
 *interchangeable.
 *
 * \section sec Prerequisites
 *
 * HyPE depends on the following third-party libraries and tools:
 *   - <a href="http://www.boost.org/">boost::filesystem, boost::system,
 *boost::thread, boost::program_options, boost::chrono</a>
 *   - Headers of the "C++ Technical Report on Standard Library Extensions", or
 *boost::tr1
 *   - <a href="http://loki-lib.sourceforge.net/">Loki Library</a>
 *   - <a href="http://www.alglib.net/">ALGLIB</a> (included, usually no need
 *install separately)
 *   - <a href="http://www.stack.nl/~dimitri/doxygen/">Doxygen</a> (only if you
 *would like to build the documentation)
 *
 * \section install_linux Installing on Linux
 *
 * HyPE has been tested with the following compilers/build-environments on
 *Linux:
 *   - Ubuntu 11.04 (32-bit)
 *   - Ubuntu 12.04 (64-bit)
 *   - gcc/g++ (v4.6.3)
 *
 * To install all prerequisites on Ubuntu Linux, install the following packages:
 * @code
 * sudo apt-get install gcc g++ make cmake doxygen graphviz libboost-all-dev
 *build-essential gnuplot-x11 ffmpeg libloki-dev libloki*
 * @endcode
 *
 * Now you can configure the HyPE package.
 * We advise you to do a CMake out-of-source-tree build, for instance using the
 *following
 * commands from a shell:
 * @code
 * cd hype-library
 * mkdir build
 * cd build
 * cmake ../
 * @endcode
 * If you encounter any errors during configuration, or want to tweak HyPE's
 *build system you may want to run `ccmake`
 * to manipulate CMake's cache and re-generate the build-system.
 * The appropriate options are documented.
 *
 * To build HyPE, use:
 * @code
 * make
 * @endcode
 * To build and run the test suite, type:
 * @code
 * make check
 * @endcode
 * To build the documentation, type:
 * @code
 * make hype-doc
 * @endcode
 * To install HyPE, type (as root):
 * @code
 * make install
 * @endcode
 *
 * \section install_windows Installing on Windows
 *
 * HyPE was tested with Windows using the following toolchains:
 *   - <a href="http://www.cygwin.com/">Cygwin</a>
 *   - <a href="http://www.mingw.org/">Minimal GNU for Windows (32-bit)</a>, GCC
 *4.7.2
 *   - <a href="http://www.microsoft.com/visualstudio/">Visual Studio C++ 2010
 *Express</a> (32-bit)
 *
 * Naturally, other toolchains like MinGW-64 and newer versions of Visual Studio
 * might work as well.
 *
 * In any case, if you would like to build HyPE's documentation, install Doxygen
 *via its
 * Windows installer.
 * If you let the installer add Doxygen to `PATH`, the build system will locate
 * it automatically.
 *
 * You must also install CMake, presumably via its Windows installer.
 * Let the installer add CMake to `PATH` as well.
 *
 * \subsection install_mingw32 Installing with Minimal GNU for Windows
 *
 * To install HyPE using the MinGW toolchain, first build MinGW versions of
 * the dependant libraries.
 * They do not necessarily have to be installed into the MinGW/MSYS path
 *hierarchy.
 *
 * Boost <b>must</b> be built for the MinGW toolchain, Visual Studio builds will
 * not work.
 * If you cannot find an appropriate binary, build the binaries from a MinGW
 *command shell
 * (cmd.exe):
 * @code
 * cd boost_1_53_0
 * boostrap mingw
 * bjam toolset=gcc
 * @endcode
 *
 * The Loki library <b>must</b> also be built for the MinGW toolchain.
 * At first you must remove Loki's `src/LevelMutex.cpp`, since it is broken on
 *MinGW and
 * not required by HyPE.
 * From a MinGW shell (MSYS Bash), type:
 * @code
 * cd loki-0.1.7/
 * mingw32-make build-static build-shared OS=Windows
 * @endcode
 *
 * Now you are ready to build HyPE.
 * From a MSYS Shell, type:
 * @code
 * cd hype-library
 * mkdir build
 * cd build
 * cmake-gui ../
 * @endcode
 * Choose "MSYS Makefiles" as the build system to generate and
 * click _Configure_ to configure the HyPE package.
 * There will be errors.
 * In the CMake cache
 *   - set `BOOST_ROOT` to the build location of Boost for MinGW
 *   - configure the `Boost_USE_` options; presumably enable
 *`Boost_USE_STATIC_LIBS`
 *
 * Now _Configure_ again - Boost should be properly configured now but not the
 *Loki library.
 * So in the cache (advanced), set
 *   - `Loki_INCLUDE_DIRS` to the `include/` subdirectory of Loki (e.g.
 *`C:/loki-0.1.7/include`)
 *   - `Loki_LIBRARIES` to the path of Loki's static library (`libloki.a`) or
 *DLL (`libloki.dll`)
 *
 * HyPE should now <b>Configure</b> properly and you can click <b>Generate</b>.
 *
 * From a MSYS Shell you may now build HyPE just like you would on Linux:
 * @code
 * cd hype-library/build
 * make
 * @endcode
 * It may also be installed into the MinGW/MSYS paths.<br/>
 * <b>Note</b>: In order to run the test suite, copy all necessary DLLs to
 * `hype-library/build/examples/unitests/`.
 *
 * \subsection install_msvc Installing with Visual Studio C++
 *
 * To install HyPE using the Visual Studio C++ toolchain, first build MSVC
 *versions of
 * the dependant libraries.
 *
 * Boost <b>must</b> be built for the MSVC toolchain, MinGW builds will
 * not work.
 * If you cannot find an appropriate binary, build the sources from Windows
 *command shell
 * (cmd.exe):
 * @code
 * cd boost_1_53_0
 * boostrap
 * bjam
 * @endcode
 *
 * The Loki library <b>must</b> also be built for the MSVC toolchain.
 * From a Windows shell (cmd.exe) using Visual Studio C++ 2010, type:
 * @code
 * cd loki-0.1.7/
 * set VS80COMNTOOLS=%VS100COMNTOOLS%
 * make.msvc.bat
 * @endcode
 *
 * Now you are ready to build HyPE.
 * First generate the Visual Studio project files using CMake.
 * To do so, start CMake GUI and select the HyPE source directory.
 * You may select a different build directory for out-of-source-tree builds.
 * Click _Configure_ to configure the HyPE package.
 * There will be errors.
 * In the CMake cache
 *   - set `BOOST_ROOT` to the build location of Boost for MSVC
 *   - configure the `Boost_USE_` options; presumably enable
 *`Boost_USE_STATIC_LIBS`
 *
 * Now _Configure_ again - Boost should be properly configured now but not the
 *Loki library.
 * So in the cache (advanced), set
 *   - `Loki_INCLUDE_DIRS` to the `include/` subdirectory of Loki (e.g.
 *`C:/loki-0.1.7/include`)
 *   - `Loki_LIBRARIES` to the path of Loki's MSVC static library (`loki.lib`)
 *
 * HyPE should now <b>Configure</b> properly and you can click <b>Generate</b>.
 *
 * In Visual Studio, you can now open the generated solution file `HyPE.sln`
 *(located in the
 * build directory) and build the entire solution or a specific target.<br/>
 * <b>Note</b>: In order to run the test suite, copy all necessary DLLs to
 * `hype-library/build/examples/unitests/` (or whatever your build directory
 *is).
 *
 * <b>Note also</b>, that when performing a _Debug_ build, all dependent
 *libraries must be
 * debug versions as well.
 * Debug versions of Boost are automatically selected by the build system, but
 *Loki library
 * Debug versions must be manually built and selected in the CMake cache.
 */

/*! \page page_tutorial Tutorial: How to use HyPE
  \tableofcontents
  HyPE is organized as a library to allow easy integration in existing
applications. You can choose between a dynamic and a static version of the
library. Note that you have to link against the libraries HyPE uses, if you use
the static version.
   *
        * <b>ATTENTION: HyPE uses the thread library of boost for it's advanced
features. There is a bug concerning applications compiled with the g++ compiler,
because boost thread does not properly export all symbols until version 1.48.
        * The Bug was fixed in <a
href="http://www.boost.org/users/history/version_1_49_0.html">Boost 1.49</a>.
        * The workaround is to statically link against boost thread. Further
details can be found <a
href="https://svn.boost.org/trac/boost/ticket/2309">here</a>.</b>
        *
        * To integrate HyPE in your project, you have to include the header file
        *  @code
#include <hype.hpp>
        *	@endcode
        *  and link against hype:
        *  @code
g++ -g  -Wl,-rpath,${PATH_TO_HYPE_LIB}/lib -Wall -Werror -o <you application's
name> <object files> -I${PATH_TO_HYPE_LIB}/include -Bstatic -lboost_thread
-pthread -Bdynamic -L${PATH_TO_HYPE_LIB}/lib -lhype -lboost_system
-lboost_filesystem -lboost_program_options-mt -lloki -lrt
        *	@endcode

         * \subsection sec_api_usage_hype  Use the API
         *
         * The general concept of HyPE is to make decisions for your
applications, which algorithm (processing device) should be used to perform an
operation.
         * Therefore, you first have to specify the Operations you wish to make
decisions for and second you have to register your available algorithms for
these operations.
         * First we need a reference to the global Scheduler:
         * @code
hype::Scheduler& scheduler=hype::Scheduler::instance();
        *	@endcode

         * HyPE uses two major abstractions: First, a DeviceSpecification, which
defines information to a processing device, e.g., a CPU or GPU.
         * Second, is an AlgorithmSpecification, which encapsulates algorithm
specific information, e.g., the name, the name of the operation
         * the algorithm belongs to as well as the learning and the load
adaption strategy.
         *
         * As example, we will create the configuration for the most common
case: A system with one CPU and one dedicated GPU:
         *
         * @code
DeviceSpecification cpu_dev_spec(hype::PD0, //by convention, the first CPU has
Device ID: PD0  (any system has at least one)
                                                                                        hype::CPU, //a CPU is from type CPU
                                                                                        hype::PD_Memory_0); //by convention, the host main memory has ID PD_Memory_0

DeviceSpecification gpu_dev_spec(hype::PD1, //different porcessing device
(naturally)
                                                                                        hype::GPU, //Device Type
                                                                                        hype::PD_Memory_1); //separate device memory
        *	@endcode
         *
         * Now, we have to define the algorithms. Note that an algorithm may
utilize only one processeng device at a time (e.g., the GPU).
         *
         * @code
AlgorithmSpecification cpu_alg("CPU_Algorithm",
                                                                                 "SORT",
                                                                                 hype::StatisticalMethods::Least_Squares_1D,
                                                                                 hype::RecomputationHeuristics::Periodic,
                                                                                 hype::OptimizationCriterions::ResponseTime);

AlgorithmSpecification gpu_alg("GPU_Algorithm",
                                                                                 "SORT",
                                                                                 StatisticalMethods::Least_Squares_1D,
                                                                                 RecomputationHeuristics::Periodic,
                                                                                 OptimizationCriterions::ResponseTime);
        *	@endcode
 *
 * Note that the GPU algorithm is only executable on the GPU and hence, should
be assigned only to
 * DeviceSpecifcations of ProcessingDeviceType GPU. <b>ATTENTION: the algorithm
name in the
 * AlgorithmSpecification has to be unique!</b>
 * Let's assume that our CPU algorithm runs only
 * on the CPU and the GPU algorithms runs only on the GPU. We define this by
calling the method Scheduler::addAlgorithm:

         * @code
scheduler.addAlgorithm(cpu_alg, cpu_dev_spec); //add CPU Algorithm to CPU
Processing Device
scheduler.addAlgorithm(gpu_alg, gpu_dev_spec); //add GPU Algorithm to GPU
Processing Device
        *	@endcode
         *
         * We are now ready to use the scheduling functionality of HyPE.
         * First, we have to identify the parameters of a data set that have a
high impact on the algorithms execution time.
         * In case of our sorting example, we identify the size of the input
array as the most important feature value. Note that HyPE supports n feature
values (n>=1).
         *
         * To tell HyPE the feature value(s) of the data set that is to be
processed, we have to store them in a hype::Tuple object. By convention, the
first entry
         * quantifies the size of the input data, and the second (if any) should
contain the selectivity of a database operator.
         * @code
hype::Tuple t;
t.push_back(Size_of_Input_Dataset);//for our sort operation, we only need the
data size
        *	@endcode
         *
         * Now, HyPE knows about your hardware and your algorithms. We can now
let HyPE do scheduling decisions.
         * HyPE needs two informations to perform scheduling decisions: First is
a OperatorSpecification, which
         * defines the operation that should be executed ("SORT"), and the
feature vector of the input data (t).
         * Furthermore, we have to specify the location of the input data as
well as the desired location for
         * the output data, so HyPE can take the cost for possible copy
operations into account.
         *
         * @code
OperatorSpecification op_spec("SORT",
                                                                                t,
                                                                                hype::PD_Memory_0, //input data is in CPU RAM
                                                                                hype::PD_Memory_0); //output data has to be stored in CPU RAM
        *	@endcode
         * The second information, which HyPE needs, is a specification of
constraints on the processing devices.
         * For some applications, operations cannot be executed on all
processing devices for arbitrary data. For example,
         * if a GPU has not enough memory to process a data set, the operation
will fail (and will probably slow down
         * other operations). Since HyPE cannot know this, because it does not
know the semantic of the operations, the user
         * can specify constraints, on which type of processing device the
operation may be executed.
         * In our case, we have no constraints and just default construct a
DeviceConstraint object.
         * @code
DeviceConstraint dev_constr;
         *	@endcode
         *
         * No we can ask HyPE were to execute our operation:
         * @code
SchedulingDecision sched_dec = scheduler.getOptimalAlgorithm(op_spec,
dev_constr);
         * @endcode
         *
        *  Note that the application <b>always</b> has to execute the algorithm
HyPE chooses, otherwise, all
        *  following calls to sched_dec.getOptimalAlgorithm() will have
undefined behavior.
        *  Since HyPE uses a feedback loop to refine the estimations of
algorithm execution times, you have
        *  to measure the execution times of your algorithms and pass them back
to HyPE.
        *  HyPE provides a high level interface for algorithm measurement:
        *  @code
AlgorithmMeasurement alg_measure(sched_dec); //has to be directly before
algorithm execution
        //execute the choosen algortihm
alg_measure.afterAlgorithmExecution();  //has to be directly after algorithm
termination
        *	@endcode
         * The AlgorithmMeasurement object starts a timer and
afterAlgorithmExecution() stops the timer. Note that the constructor of the
AlgorithmMeasurement object needs a SchedulingDecision as parameter.
         * When we put the usage of the SchedulingDecision together, we get the
following code skeleton:
         * @code
if(sched_dec.getNameofChoosenAlgorithm()=="CPU_Algorithm"){
        AlgorithmMeasurement alg_measure(sched_dec);
                //execute "CPU_Algorithm"
        alg_measure.afterAlgorithmExecution();
}else if(sched_dec.getNameofChoosenAlgorithm()=="GPU_Algorithm"){
        AlgorithmMeasurement alg_measure(sched_dec);
                //execute "GPU_Algorithm"
        alg_measure.afterAlgorithmExecution();
}
          * @endcode

           * Some applications have their own time measurement routines and wish
to use their own timer framework. To support such applications,
                * HyPE offers a direct way to add a measured execution time in
nanoseconds for a corresponding SchedulingDecision:
           * @code
uint64_t begin=hype::core::getTimestamp();
CPU_algorithm(t[0]);
uint64_t end=hype::core::getTimestamp();
//scheduling decision and measured execution time in nanoseconds!!!
scheduler.addObservation(sched_dec,end-begin);
           * @endcode

           * The complete source code of this example can be found in the
documentation \ref online_learning.cpp and in the examples directory of HyPE
(examples/use_as_online_framework/online_learning.cpp).
                *
 * */

/*! \page page_existing_plugins_hype Available Components
  \tableofcontents

  \section sec_statistical_method Statistical Methods

         A Statistical Method learns the relation between the feature values of
 the input dataset and an algorithms execution time. It is a central part of
 HyPE, implementing the learning based execution time estimation.
          * Hence, it is crucial to select the appropriate statistical method
 depending on the algorithm and the application environment.
          * Currently, HyPE supports one dimensional Least Square Method and
 Multi Linear Fitting. Statistical methods are defined in the type
 hype::StatisticalMethods.
          *
                \subsection subsec_statistical_method_leastsquares Least Square
 Method
                HyPE uses the least square solver of the ALGLIB Project. It is
 usually the candidate to choose, if an algorithm only depends on one input
 features, such as sorting.

                \subsection subsec_statistical_method_multi_linear_fitting Multi
 Linear Fitting
                HyPE uses the multi linear fitting functionality of the ALGLIB
 Project. You should choose Multi Linear Fitting, if an algorithm depends on
 multiple input features, such as selections (data size, selectivity).
                <b>Note that Multi Linear Fitting is currently limited to two
 features, but will support more in future versions of HyPE.</b>

  \section sec_recomputation_heuristic Recomputation Heuristics
   A Recomputation Heuristic implements the load adaption functionality of HyPE.
 If the load situation of a system dramatically changes, then it is very likely
 that the execution time of algorithm will change as well.
         * To ensure sufficiently exact estimations, the learned approximation
 functions have to be updated. However, there is now 'perfect' point in time
 when to recompute the approximation functions.
         * Therefore, the user can select a Recomputation Heuristic, which is
 appropriate for the application. Recomputation heuristics are defined in the
 type hype::RecomputationHeuristics.
         *
         \subsection subsec_recomputation_heuristic_periodic Periodic
 Recomputation
          * The Periodic Recomputation Heuristic will recompute the
 approximation function of an algorithm after X executions of this algorithm. X
 is called Recomputation
          * Period and can be configured as well (see \ref page_configure_hype
 for details).
          * You should use this Recomputation Heuristic if you want that HyPE
 refines its estimations at runtime to adapt at changing data, load, etc.

         \subsection subsec_recomputation_heuristic_oneshot Oneshot
 Recomputation
          * The Oneshot Recomputation Heuristik will compute the approximation
 functions once after the initial training phase. You should choose this
 optimization criterion, if significant changes in the load in your system is
 seldom or have little impact on algorithms execution time.

         \subsection subsec_recomputation_heuristic_error_based Error based
 Recomputation (under development)

  \section sec_optimization_criterion Optimization Criterions
   An Optimization Criterion specifies what an "optimal" algorithm for your
 application is. Should it be the fastest?
        Or would you like to select algorithms in a way that the throughput of
 your system is optimized?
        Therefore, we implemented several strategies to make HyPE configurable
 and better usable for a wide range of applications.
        Optimization criteria are defined in the type
 hype::OptimizationCriterions.
         \subsection subsec_optimization_criterion_response_time Response Time
          The idea of Response Time optimization is to reduce the execution time
 of one operation by selecting the (estimated) fastest algorithm.

         \subsection subsec_optimization_criterion_WTAR Waiting Time Aware
 Response Time
          Waiting Time Aware Response Time (WTAR) is an extension of the simple
 response time algorithm. WTAR takes into account the load
          on all processing devices and allocates for an operation O the
 processing device, were the sum of the waiting time,
          until the previous oeprators finished, and the estimated execution
 time of O is minimal. This is the recommended optimization algorithm for HyPE.

         \subsection subsec_optimization_criterion_RR Round Robin
          * The round robin strategy allocates processing devices for operations
 in turns, distributing a workload of operations on all processing devices.
          * This approach works well in case operation need roughly the same
 time on all processing devices (e.g., on homogeneous hardware).
          * However, in case one processing device is significantly faster than
 the other processing devices, the round strategy will under utilize the
          * faster processing device, and over utilize the slower processing
 devices.

 *  \subsection subsec_optimization_criterion_TBO Threshold-based Outsourcing
 *  Threshold-based Outsourcing is an extension of Response Time. The idea is to
 force the use of a slower processing device
 *  to relieve the fastest processing device and distribute the workload on all
 available processing devices. However, the algorithm has to ensure
 *  that the operation's response time does not significantly increase.
 Therefore, an operation may be executed on a slower processing device,
 *  if and only if the expected slowdown is under a certain threshold W.
 *
 *  \subsection subsec_optimization_criterion_PBO Probability-based Outsourcing
 *
 * Probability-based Outsourcing computes for each scheduling decision the
 estimated execution times of the avaialble algorithms. Then, each algorithm
 gets
 * assigned a probability, depending on the estimated execution time. Faster
 algorithms (on faster processing devices) get a higher probability to be
 executed
 * then slower algorithms. Depending on the probability, an algorithm is chosen
 randomly for execution.
 *
          *
          *
*/

//"Simple Round Robin"
//"Response Time"
//"WaitingTimeAwareResponseTime"
//"Throughput"
//"ProbabilityBasedOutsourcing"

//	if(!scheduler.setOptimizationCriterion("SORT","Simple Round Robin"))
//		std::cout << "Error: Could not set Optimization Criterion!" <<
// std::endl;	else cout << "Success..." << endl;

//	if(!scheduler.setOptimizationCriterion("SORT","Response Time"))
//		std::cout << "Error: Could not set Optimization Criterion!" <<
// std::endl;	else cout << "Success..." << endl;

//	if(!scheduler.setOptimizationCriterion("SORT","WaitingTimeAwareResponseTime"))
//		std::cout << "Error: Could not set Optimization Criterion!" <<
// std::endl;	else cout << "Success..." << endl;

//	if(!scheduler.setOptimizationCriterion("SORT","Throughput"))
//		std::cout << "Error: Could not set Optimization Criterion!" <<
// std::endl;	else cout << "Success..." << endl;

//	if(!scheduler.setOptimizationCriterion("SORT","ProbabilityBasedOutsourcing"))
//		std::cout << "Error: Could not set Optimization Criterion!" <<
// std::endl;	else cout << "Success..." << endl;

/*! \page page_configure_hype Configure HyPE
  \tableofcontents

   * HyPE can be configured in four ways. First, modify the Static_Configuration
 of HyPE, which sets default values for all variables.
        * Second, update the configuration at runtime. The class
 Runtime_Configuration provides methods to change all modifiable variables.
        * Note that not all variables are modifiable during runtime.
        * Third, create a configuration file 'hype.conf', and add the variables
 with their corresponding values.
        * Note, that the structure of the file for each line is
 variable_name=value and one line may at most contain one assignment.
        * Fourth, specify parameter values in environment variables.
        *

        * - modify the hype::core::Static_Configuration of HyPE (requires
 recompilation)
        * - update the configuration at runtime using
 hype::core::Runtime_Configuration
        * - create a configuration file 'hype.conf', and add the variables with
 their corresponding values
        * 		- <b>help</b>                        produce help
 message
        * 		- <b>length_of_trainingphase</b> set the number
 algorithms executions to complete training
        * 		- <b>history_length</b> set the number of measurement
 pairs that are kept in the history (important for precision of approximation
 functions)
        * 		- <b>recomputation_period</b> set the number of
 algorithm executions to trigger recomputation
        * 		- <b>algorithm_maximal_idle_time</b> set maximal
 number of operation executions, where an algorithm was not executed; forces
 retraining of algorithm
        * 		- <b>retraining_length</b> set the number of algorithm
 executions needed to complete a retraining phase (load adaption feature)
        * 		- <b>ready_queue_length</b> set the number of
 operators that are queued on a processing device, before scheduling decision
 stops scheduling
        * 			new operators (The idea is to wait how the
 done scheduling decisions turn out and to adjust the scheduling accordingly)
 * 	* 		- <b>print_algorithm_statistics</b> set the mode for
 storing algorithm statistics, true means algorithms statistics are dumped to
 the output directory of HyPE, false disables the feature (default)
        * - specify parameter values in environment variables:
        * 		- HYPE_LENGTH_OF_TRAININGPHASE set the number
 algorithms executions to complete training
        * 		- HYPE_HISTORY_LENGTH set the number of measurement
 pairs that are kept in the history (important for precision of approximation
 functions)
        * 		- HYPE_RECOMPUTATION_PERIOD set the number of
 algorithm executions needed to complete a retraining phase (load adaption
 feature)
        *		- HYPE_ALGORITHM_MAXIMAL_IDLE_TIME set maximal number
 of operation executions, where an algorithm was not executed; forces retraining
 of algorithm
        * 		- HYPE_RETRAINING_LENGTH set the number of algorithm
 executions to trigger recomputation
        * 		- HYPE_READY_QUEUE_LENGTH set the number of operators
 that are queued on a processing device, before scheduling decision stops
 scheduling
        * 			new operators (The idea is to wait how the
 done scheduling decisions turn out and to adjust the scheduling accordingly)
        * 		- HYPE_PRINT_ALGORITHM_STATISTICS set the mode for
 storing algorithm statistics, true means algorithms statistics are dumped to
 the output directory of HyPE, false disables the feature (default)
        *
        *
*/

/* \page page_structure_hype Structure of HyPE
        *
        * HyPE is structured to be a DBMS independent library, meaning templates
  *and functions can be used without modifcation.
        * However, to acheive this goal, the user has to implement a mapping
  *layer, between HyPE and the hybrid DBMS:
        *
        * \htmlonly
        * <p><img src="architecture.svg" align="center" width="40%"></p>
        * \endhtmlonly
        *
        *
        *
        *
        *
*/

/*! \page page_extending_hype Extend HyPE
  \tableofcontents


 *    \section sec_statistical_method Statistical Methods
 * HyPE learns the correlation between features of the input data set and the
resulting execution time
 * of an algorithm on a specific processing device. To allow the user to fine
tune the statistical method,
 * HyPE provides a plug-in architecture, where the user can choose either from
the set of implemented
 * statistical methods, or alternatively, implement and integrate the preferred
statistical method in HyPE.
 *
 * To create a new statistical method, the user has to inherit from the abstract
base class
 * hype::core::StatisticalMethod and implement its pure virtual methods.
 * Since HyPE uses a plug-in architecture based on factories, a static member
function <b>create</b> should
 * be defined, which returns a pointer to a new instance of your new statistical
method (e.g., Least_Squares_Method_1D).
           * @code
#include <core/statistical_method.hpp>

namespace hype{
        namespace core{

                class Least_Squares_Method_1D : public StatisticalMethod {
                public:
                        Least_Squares_Method_1D();

                        virtual const EstimatedTime computeEstimation(const
Tuple& input_values);

                        virtual bool recomuteApproximationFunction(Algorithm&
algorithm);

                        virtual bool inTrainingPhase() const throw();

                        virtual void retrain();

                        static Least_Squares_Method_1D* create(){
                                return new Least_Squares_Method_1D();
                        }

                        virtual ~Least_Squares_Method_1D();
                };
        }  //end namespace core
}  //end namespace hype
           * @endcode

        * After the user added the class, he needs to extend the enumeration
hype::StatisticalMethods::StatisticalMethod
        * in file global_definitions.hpp by a new member, which identifies the
plug-in. Finally, the user has to register the
        * plug-in in the class hype::core::PluginLoader in file
pluginloader.cpp.
 *
   *   \section sec_recomputation_heuristic Recomputation Heuristics
 * HyPE is capable of refining estimated execution times at run-time. Depending
on the application, a run-time refinement is
 * beneficial or causes only additional overhead. To support a wide range of
applications, HyPE allows to fine tune
 * the runtime refinement on a per algorithm basis (e.g, for the same operation,
we can have runtime adaption on the CPU,
 * but not on the GPU.)
 *
 * HyPE provides a plug-in architecture, where the user can choose either from
the set of implemented
 * recomputation heuristics, or alternatively, implement and integrate the
preferred recomputation heuristic in HyPE.
 *
 * To create a new recomputation heuristic, the user has to inherit from the
abstract base class
 * hype::core::RecomputationHeuristic and implement its pure virtual methods.
 * Since HyPE uses a plug-in architecture based on factories, a static member
function <b>create</b> should
 * be defined, which returns a pointer to a new instance of your new
recomputation heuristic (e.g., Oneshotcomputation).
 *
        * @code
#include <core/recomputation_heuristic.hpp>

namespace hype{
        namespace core{

                class Oneshotcomputation : public RecomputationHeuristic {
                        public:
                        Oneshotcomputation();
                        //returns true, if approximation function has to be
recomputed and false otherwise
                        virtual bool internal_recompute(Algorithm& algortihm);

                        static Oneshotcomputation* create(){
                                return new Oneshotcomputation();
                        }

                };
        }  //end namespace core
}  //end namespace hype
        * @endcode
        * After the user added the class, he needs to extend the enumeration
hype::RecomputationHeuristics::RecomputationHeuristic
        * in file global_definitions.hpp by a new member, which identifies the
plug-in. Finally, the user has to register the
        * plug-in in the class hype::core::PluginLoader in file
pluginloader.cpp.


        *   \section sec_optimization_criterion Optimization Criterions
        *
        * 	 To add a new optimization criterion, the user has to inherit
from the abstract class hype::core::OptimizationCriterion and implement its pure
vitual methods.
        * 	 Since HyPE uses a plug-in architecture based on factories, a
static member function <b>create</b> should be defined, which returns a pointer
to a new instance of
        *   your new optimization criterion, which we will call
"NewResponseTime".

           * @code
#include <core/optimization_criterion.hpp>

namespace hype{
        namespace core{
                class NewResponseTime : public OptimizationCriterion{
                        public:
                        NewResponseTime(const std::string& name_of_operation);

                        virtual const SchedulingDecision
getOptimalAlgorithm_internal(const Tuple& input_values, Operation& op,
DeviceTypeConstraint dev_constr);
                        //factory function
                        static NewResponseTime* create(){
                                return new NewResponseTime("");
                   }
                }
        };
};
           * @endcode
        *
        * After the user added the class, he needs to extend the enumeration
hype::OptimizationCriterions::OptimizationCriterion
        * in file global_definitions.hpp by a new member, which identifies the
plug-in. Finally, the user has to register the
        * plug-in in the the class hype::core::PluginLoader in file
pluginloader.cpp.
        *
*/

/*! \page page_faq FAQ
   *
        *  -# <b>What is HyPE?</b><br>
        *    HyPE is a Hybrid query Processing Engine build for automatic
  *selection of processing units for co-processing in database systems. The
  *long-term goal of the project is to implement a fully fledged query
  *processing engine, which is able to automatically generate and optimize a
  *hybrid CPU/GPU physical query plan from a logical query plan.
        *  -# <b>When should I use HyPE?</b><br>
        *  You can use HyPE whenever you want to decide on a CPU and a GPU
  *implementation of an operation in your application at run-time. In other
  *words, any GPU accelerated application can make use of hype to effectively
  *utilize existing processing ressources.
        *  -# <b>Under which license is HyPE distributed?</b><br>
        *  HyPE is licenced under the <a
  *href="http://www.gnu.org/licenses/lgpl-3.0.txt">GNU LESSER GENERAL PUBLIC
  *LICENSE - Version 3</a>.
        *  -# <b>Which platforms are currently supported?</b><br>
        * 	HyPE compiles and runs under Linux and Windows (Cygwin). Native
  *Windows support is planned for future releases.
        *  -# <b>I have a problem in using HyPE, how can I get help?</b><br>
        *  For information about the project, technical questions and bug
  *reports: please contact the development team via <a
  *href="http://wwwiti.cs.uni-magdeburg.de/~bress/">Sebastian\htmlonly
  *Bre&szlig;\endhtmlonly\latexonly Bre{\ss}\endlatexonly</a>.
        *
*/
