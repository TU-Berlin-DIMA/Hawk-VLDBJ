/***********************************************************************************************************
Copyright (c) 2013, Sebastian Breß, Otto-von-Guericke University of Magdeburg, Germany. All rights reserved.

This program and accompanying materials are made available under the terms of the 
GNU GENERAL PUBLIC LICENSE - Version 3, http://www.gnu.org/licenses/gpl-3.0.txt
 **********************************************************************************************************/

/*!
 *  \file documentation.hpp    
 *  \brief This file contains additional documentation, like the generated web pages in the doxygen documentation.
 *  \author    Sebastian Breß
 *  \version   0.1
 *  \date      2013
 *  \copyright GNU GENERAL PUBLIC LICENSE - Version 3, http://www.gnu.org/licenses/gpl-3.0.txt
 */


/*! \mainpage Documentation
 *
 * \section intro_sec Introduction
 *
 *
 *%CoGaDB is a prototype of a column-oriented GPU-accelerated database management system developed at the University of Magdeburg. Its purpose is to investigate advanced coprocessing techniques for effective GPUs utilization during database query processing. It utilizes our hybrid query processing engine (HyPE) for the physical optimization process.
 *
 *%CoGaDB's main purpose is to investigate a GPU-aware database architecture to achieve optimal performance of DBMS on hybrid CPU/GPU platforms. We are currently working on a architecture proposal and try to benefit from past experiences of hybrid CPU/GPU DBMS. Therefore, %CoGaDB provides an extensible architecture to enable researchers an easy integration of their GPU-accelerated operators, coprocessing techniques and query optimization heuristics. Note that %CoGaDB assumes that the complete database can be kept in main memory, because GPU-acceleration is not beneficial for workloads where disc I/O is the dominating factor.
 *
 * 	\section features_sec Features
 * 
Currently, CoGaDB implements the following features:
 *    - Written mainly in C++ and Cuda C
 *    - Column-oriented in-memory database management system
 *    - SQL Interface
 *    - CPU and GPU operators for selection, sort, join, and simple aggregations using optimized parallel algorithms from the libraries Intel® TBB (CPU) and Thrust (GPU) and CPU only operators for projections and other management operators
 *    - Uses HyPE, our hybrid query processing engine, for physical optimization and query processing \latexonly\citep{Bress13,BressBR+13}\endlatexonly
 *    - Capable of data compression:
 *       - Run Length Encoding
 *       - Bit Vector Encoding
 *       - Dictionary Compression
 *       - Delta Coding
 *    - NEW: SIMD Scan
 *    - NEW: Supports filtering of strings on the GPU
 *    - NEW: Support for primary key and foreign key integrity constraints
 *
 * 
 * 	\section platforms_sec Supported Platforms
  *    - Runs (currently) only on Linux: Ubuntu 12.04 (32 and 64 Bit)
 * 
 * 
 * 	\section documentation_sec Detailed Documentation
 *
 *	Here a list for more detailed documentation:
 *    - \subpage page_install_cogadb
 *    - \subpage page_using_cogadb
 *    - \subpage page_structure_cogadb 
 *    - \subpage page_concepts_in_cogadb
 *    - \subpage page_faq 
 * 
 * 
 */

/*
 * \section members_sec Project
 * 
 * \subsection members_subsec Project Members:
 * - <a href="http://wwwiti.cs.uni-magdeburg.de/~bress/">Sebastian\htmlonly Bre&szlig;\endhtmlonly\latexonly Bre{\ss}\endlatexonly</a> (University of Magdeburg)
 * - Robin Haberkorn (University of Magdeburg)
 * - Steven Ladewig (University of Magdeburg)
 * - Tobias Lauer (Jedox AG)
 * - Manh Lan Nguyen (University of Magdeburg)
 * - <a href="http://wwwiti.cs.uni-magdeburg.de/~saake/">Gunter Saake</a> (University of Magdeburg)
 * - <a href="http://www.infosun.fim.uni-passau.de/spl/people-nsiegmund.php">Norbert Siegmund</a> (University of Passau)
 * \subsection contributors_subsec Contributors:
 * - Darius Brückers (contributed Compression Technique: Run Length Encoding)
 * - Sebastian Krieter (contributed Compression Technique: Delta Coding)
 * - Steffen Schulze (contributed Compression Technique: Bit Vector Encoding) 
 * \subsection partners_subsec Project Partners:
 * - Ladjel Bellatreche (LIAS/ISEA-ENSMA, Futuroscope, France)
 * \subsection former_members_subsec Former Project Members:
 * - René Hoyer (University of Magdeburg)
 * - Ingolf Geist (University of Magdeburg)
 * - Patrick Sulkowski (University of Magdeburg)
 * 
 * 
 * \section publication_sec Publications 
\htmlonly
<ul class="bib2xhtml">

<!-- Authors: Sebastian Bress -->
<li><a name="thesisBress"></a>Sebastian Bre&szlig;.
<a href="http://wwwiti.cs.uni-magdeburg.de/iti_db/publikationen/ps/auto/thesisBress.pdf">Ein selbstlernendes Entscheidungsmodell f&uuml;r die Verteilung von
  Datenbankoperationen auf CPU/GPU-Systemen</a>.
Master thesis, University of Magdeburg, Germany, March 2012.
In German.</li>

<!-- Authors: Sebastian Bress and Siba Mohammad and Eike Schallehn -->
<li><a name="BressMS12"></a>Sebastian Bre&szlig;, Siba Mohammad, and
  Eike Schallehn.
<a href="http://wwwiti.cs.uni-magdeburg.de/iti_db/publikationen/ps/auto/bress:2012:GvDB:decision_model.pdf">Self-Tuning Distribution of DB-Operations on
  Hybrid CPU/GPU Platforms</a>.
In <cite>Proceedings of the 24st Workshop Grundlagen von Datenbanken
  (GvD)</cite>, pages 89&ndash;94. CEUR-WS, 2012.</li>

<!-- Authors: Sebastian Bress and Eike Schallehn and Ingolf Geist -->
<li><a name="BressSG12"></a>Sebastian Bre&szlig;, Eike Schallehn, and
  Ingolf Geist.
<a href="http://link.springer.com/chapter/10.1007/978-3-642-32518-2_3">Towards
  Optimization of Hybrid CPU/GPU Query Plans in Database Systems</a>.
In <cite>Second ADBIS workshop on GPUs In Databases (GID)</cite>, pages 27&ndash;35.
  Springer, 2012.</li>

<!-- Authors: Sebastian Bress and Felix Beier and Hannes Rauhe and Eike
  Schallehn and Kai Uwe Sattler and Gunter Saake -->
<li><a name="BressBR+12"></a>Sebastian Bre&szlig;, Felix Beier, Hannes
  Rauhe, Eike Schallehn, Kai-Uwe Sattler, and Gunter Saake.
<a href="http://link.springer.com/chapter/10.1007/978-3-642-33074-2_5">Automatic Selection of Processing Units for Coprocessing in Databases</a>.
In <cite>16th East-European Conference on Advances in Databases and Information
  Systems (ADBIS)</cite>, pages 57&ndash;70. Springer, 2012.</li>

<!-- Authors: Sebastian Bress and Ingolf Geist and Eike Schallehn and Maik Mory
  and Gunter Saake -->
<li><a name="BressGS+13"></a>Sebastian Bre&szlig;, Ingolf Geist, Eike
  Schallehn, Maik Mory, and Gunter Saake.
<a href="http://control.ibspan.waw.pl:3000/contents/export?filename=Bress-et-al.pdf">A Framework for Cost based Optimization of Hybrid CPU/GPU Query Plans in
  Database Systems</a>.
<cite>Control and Cybernetics</cite>, 41(4):715&ndash;742,
  2012.</li>



<!-- Authors: Sebastian Bress and Stefan Kiltz and Martin Schaler -->
<li><a name="BressKS13"></a>Sebastian Bre&szlig;, Stefan Kiltz, and
  Martin Sch&auml;ler.
<a href="http://www.btw-2013.de/proceedings/Forensics%20on%20GPU%20Coprocessing%20in%20Databases%20%20Research%20Challenges%20First%20Experiments%20and%20Countermeasures.pdf">Forensics on GPU Coprocessing in Databases – Research
  Challenges, First Experiments, and Countermeasures</a>.
In <cite>Workshop on Databases in Biometrics, Forensics and Security
  Applications (DBforBFS)</cite>, BTW-Workshops, pages 115&ndash;130.
  K&ouml;llen-Verlag, 2013.</li>

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
<a href="http://link.springer.com/chapter/10.1007/978-3-319-01863-8_25">Exploring the Design Space of a GPU-aware Database Architecture</a>.
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
 * 
*/

/*! \page page_install_cogadb Installation
  \tableofcontents
   * 
	*   \section sec Installation on Ubuntu
  Just type the following command line: <br>
  @code
  sudo apt-get install gcc g++ make cmake doxygen doxygen-gui graphviz libboost-all-dev libtbb-dev libreadline6 libreadline6-dev bison
  @endcode
  Alternatively, you use our installation script:
  @code 
  ./setup-ubuntu.sh  
  @endcode
	* 
   * \section install_sec Installation
 * Currently only Linux is officially supported. For Installation, download and unpack the release package. Afterwards, you have to install the necessary tools and libraries that %CoGaDB uses:

	- <a href="http://www.boost.org/">Boost</a>: boost_filesystem, boost_system, boost_thread, boost_program_options
	- <a href="http://threadingbuildingblocks.org/">TBB</a>
	- <a href="http://cnswww.cns.cwru.edu/php/chet/readline/rltop.html">Readline</a>
	- <a href="https://developer.nvidia.com/cuda-toolkit">NVIDIA&reg; CUDA&reg; Toolkit</a>
	- <a href="http://wwwiti.cs.uni-magdeburg.de/iti_db/research/gpu/hype">HyPE Library</a>
	- <a href="http://www.gnu.org/software/bison/">Bison</a>
	 
	 We included a setup script for ubuntu users: <b>setup-ubuntu.sh</b> (Note that CUDA has to be installed separately).
  
    \subsection subsec_building_cogadb  Building cogadb in a Terminal
    
Open a terminal and navigate to the directory were you unpacked CoGaDB.  
    
  Compile cogadb
  @code
  cd gpudbms/
  mkdir build
  cd build
  cmake ../
  make
  @endcode
  
To run CoGaDB, issue

  @code
	cd build
	./cogadb/bin/cogadbd
  @endcode

  To generate and view the documentation, you can use the following commands:
	
  @code
  make cogadb-doc
  ${BROWSER} cogadb/doc/documentation/html/index.htm
  @endcode 
	*
*/


/*! \page page_using_cogadb Tutorial
  \tableofcontents
   * 

	In this section, we provide a short getting started guide for using CoGaDB. Furthermore, we list the available commands of CoGaDB's command line interface. Finally, we present a short demo of the SQL Interface to show its current capabilities.  
	*   \section getting_started_sec Getting Started
   * 
	At first, we have to create a directory, where CoGaDB can store its database:
  @code
set path_to_database=/home/DATA/coga_databases/ssb_sf1
  @endcode 

	Then, we have to create a database and import data. This can be done in two ways: using the sql interface (create table, insert into), or using a utility command.
	CoGaDB supports utility commands for importing databases from two common OLAP benchmarks: the TPC-H and the Star Schema Benchmark. Note that you have to generate the *.tbl files using the dbgen tool. Assuming we have generated a database for the star schema benchmark of scale factor one and stored the resulting *.tbl files in /home/DATA/benchmarks/star_schema_benchmark/SF1/, we can import the data with the following command: 
  @code
create_ssb_database /home/DATA/benchmarks/star_schema_benchmark/SF1/
  @endcode 
For the TPC-H benchmark, the command is create_tpch_database.

Now CoGaDB imports the data and stores them in the database. Depending on the scale factor, this can take a while. After the import finishes, we can start working with the database. Since CoGaDB is an in-memory database, we first have to load the database in the main memory:
  @code
loaddatabase
  @endcode 

Then, we can start issuing queries. We can either use SQL or build in aliases for stored queries. We provide stored queries for all queries of the star schema benchmark. The template command is ssbXY, which executes SSB-Query X.Y (X has to be a number between 1 and 4; Y has to be a number between 1 and 3 except when X is 3, in this case 4 is valid for Y as well). 

Sometimes, when no NVIDIA GPU is available, we need to restrict CoGaDB to use only the CPU. We can configure this by issuing the following command:
  @code
setdevice cpu
  @endcode 
If we want to allow CoGaDB to use both processing devices, we can replace cpu with %any. It is also possible to force the usage of the GPU for all processing tasks by specifying %gpu. However, this is NOT recommended, because for most complex queries, CoGaDB will not be able to perform all processing tasks on GPU only.

Now, we can launch queries:
  @code
CoGaDB>exec select sum(lo_extendedprice*lo_discount) as revenue from lineorder, dates where lo_orderdate = d_datekey and d_weeknuminyear = 6 and d_year = 1994 and lo_discount between 5 and 7 and lo_quantity between 26 and 35;
+-------------+
| REVENUE     | 
+=============+
| 2.49945e+10 | 
+-------------+
1 rows

Execution Time: 155.28039 ms
  @endcode 
\latexonly\newpage\endlatexonly
   *    \section scripting_sec Scripting Language
   *  CoGaDB offers a set of commands not included in SQL to ease development and debugging:
   * 

<table border="1">
<tr>
<th>Command</th>
<th>Description</th>
</tr>
<tr>
	<td>loaddatabase</td>
	<td>loads complete database in main memory</td>
</tr>
<tr>
	<td>unittests</td>
	<td>performs a self check of CoGaDB</td>
</tr>
<tr>
	<td>printschema</td>
	<td>prints the schema of the active database</td>
</tr>
<tr>
	<td>showgpucache</td>
	<td>prints status information of the GPU column cache</td>
</tr>
<tr>
	<td>simple_ssb_queries</td>
	<td>simple demonstrator for queries on SSB Benchmark data set</td>
</tr>
<tr>
	<td>set <variablename>=<variablevalue></td>
	<td>assign the value <variablevalue> to the variable <variablename></td>
</tr>
<tr>
	<td>print <variable></td>
	<td>print value of variable</td>
</tr>
<tr>
	<td>create_tpch_database <path to *.tbl files></td>
	<td>import tables of TPC-H benchmark in CoGaDB</td>
</tr>
<tr>
	<td>create_ssb_database <path to *.tbl files></td>
	<td>import tables of star schema benchmark in CoGaDB</td>
</tr>
<tr>
	<td>exec <SQL statement></td>
	<td>Execute SQL statements</td>
</tr>
<tr>
	<td>explain <SQL></td>
	<td>Display query plan generated from SQL expression</td>
</tr>
<tr>
	<td>explain_unoptimized <SQL></td>
	<td>As above, but does not apply logical optimizer before showing the plan</td>
</tr>
<tr>
	<td>hypestatus</td>
	<td>Prints all operations and corresponding algorithms registered in HyPE for CoGaDB's operators</td>
</tr>
<tr>
	<td>integrityconstraints</td>
	<td>Prints integrity constraints configured for current database</td>
</tr>
<tr>
	<td>toggleQC</td>
	<td>Toggle the state of Query Chopping activation. Per default QC is off. </td>
</tr>
<tr>
	<td>ssbXY</td>
	<td>Execute SSB-Query X.Y (X has to be a number between 1 and 4; Y has to be a number between 1 and 3 except when X is 3, in this case 4 is valid for Y as well)</td>
</tr>
<tr>
	<td>setdevice <DEVICE></td>
	<td>Sets the default device, which is used for execution. Possible values are 'cpu', 'gpu' or 'any' to use either the CPU or the GPU or both concurrently.</td>
</tr>
<tr>
	<td>setparallelizationmode <PARALLELIZATION MODE></td>
	<td>Sets the default parallelization mode for sub-plans generated during Two Phase Physical Optimization (TOPPO) in the second phase (currently only for complex selections). Valid values are 'serial' and 'parallel'</td>
</tr>
</table> 
<table border="1">
<tr>
<th>Command</th>
<th>Description</th>
</tr>
<tr>
	<td>about</td>
	<td>shows credits</td>
</tr>
<tr>
	<td>version</td>
	<td>shows version of CoGaDB</td>
</tr>
<tr>
	<td>quit</td>
	<td>exits CoGaDB</td>
</tr>
</table> 

CoGaDB has the following build in variables:

<table border="1">
<tr>
<th>Variable</th>
<th>Description</th>
</tr>
<tr>
	<td>path_to_database</td>
	<td>absolute or relative path to directory where the database is stored</td>
</tr>
<tr>
	<td>print_query_plan</td>
	<td>print the generated query plans for a SQL query (true,false)</td>
</tr>
<tr>
	<td>enable_profiling</td>	
	<td>print the query execution plan after execution with timings for each operator (true,false)</td>
</tr>
</table> 
   * 
	*   \section sql_sec SQL Interface
   * 
   *   CoGaDB supports a subset of the SQL-92 standard. We provide a short demo in the following to show the current capabilities of the SQL Interface. Note that we shortened the output of the following listings to the relevant information: query, result and execution time.

Lets first create a table:
@code
CoGaDB>exec create table Test ( id int, val varchar);
TEST:
+----+-----+
| ID | VAL | 
+====+=====+
+----+-----+
0 rows

Execution Time: 1.45447 ms
@endcode

Now we can insert data:
@code
CoGaDB>exec insert into Test values (0,'Car');
TEST:
+----+-----+
| ID | VAL | 
+====+=====+
| 0  | Car | 
+----+-----+
1 rows

Execution Time: 0.71237 ms
CoGaDB>exec insert into Test values (1,'Truck');
TEST:
+----+-------+
| ID | VAL   | 
+====+=======+
| 0  | Car   | 
| 1  | Truck | 
+----+-------+
2 rows

Execution Time: 0.32729 ms
CoGaDB>exec insert into Test values (2,'Boat');
TEST:
+----+-------+
| ID | VAL   | 
+====+=======+
| 0  | Car   | 
| 1  | Truck | 
| 2  | Boat  | 
+----+-------+
3 rows

Execution Time: 0.36719 ms
@endcode
Finally, we can query our table:
@code 
CoGaDB>exec select * from Test;
+----+-------+
| ID | VAL   | 
+====+=======+
| 0  | Car   | 
| 1  | Truck | 
| 2  | Boat  | 
+----+-------+
3 rows

Execution Time: 2.87929 ms
@endcode

Now we show a more complex query typical for OLAP workloads. We execute query 2.3 from the Star Schema Benchmark:
  @code
CoGaDB>exec select sum(lo_revenue), d_year, p_brand from lineorder, dates, part, supplier where lo_orderdate = d_datekey and lo_partkey = p_partkey and lo_suppkey = s_suppkey and p_brand= 'MFGR#2239' and s_region = 'EUROPE' group by d_year, p_brand order by d_year, p_brand;
+--------+-----------+-------------+
| D_YEAR | P_BRAND   | LO_REVENUE  | 
+========+===========+=============+
| 1992   | MFGR#2239 | 7.32066e+08 | 
| 1993   | MFGR#2239 | 6.65355e+08 | 
| 1994   | MFGR#2239 | 7.33858e+08 | 
| 1995   | MFGR#2239 | 6.22905e+08 | 
| 1996   | MFGR#2239 | 6.28615e+08 | 
| 1997   | MFGR#2239 | 7.84213e+08 | 
| 1998   | MFGR#2239 | 4.09671e+08 | 
+--------+-----------+-------------+
7 rows

  @endcode
   * 
   * 
   * 
*/

/*! \page page_structure_cogadb Architecture
  \tableofcontents
  
\latexonly
%\section sec_coga_arch_overview Overview
\section{Overview}
\label{sec_coga_arch_overview}
%
\begin{figure}
	\begin{center}
	\includegraphics[width=0.5\linewidth]{cogadb_architecture.pdf}
	\end{center}
	\caption{The architecture of CoGaDB}
	\label{fig:cogadb_architecture}
\end{figure}
We now provide an overview of CoGaDB's architecture in a top down direction. As most DBMSs, CoGaDB possesses an SQL interface that can be used to launch queries. The SQL Interface constructs an abstract syntax tree, which is then converted to a logical query plan. Then, CoGaDB's logical optimizer applies a set of optimizer rules to the logical query plan to make it more efficient (e.g., it pushes down selections and resolves cross products and implicit join conditions to joins). 

CoGaDB uses as physical optimizer our Hybrid Query Processing Engine (HyPE) \citep{Bress13,BressBR+13}. The logical plan is passed to HyPE, which has three components: a hybrid CPU/GPU optimizer, a processing device allocator and algorithm selector, and an estimation component, which estimates the execution time of an operator on a certain processing device (e.g., the CPU or the GPU). The hybrid query optimizer creates a physical query plan from a logical query plan using the algorithm selector and the cost estimator. Then, the query is executed by HyPE's execution engine. Internally, CoGaDB has to register its operators to HyPE and has to implement an adapter interface, which maps HyPE's abstract operator class to a set of functions calling the actual operators. For more information about the physical optimization phase, the interested reader is referred to the respective research papers \citep{Bress13,BressBR+13,BressGS+12}. 

Depending on the chosen processing device, data needs to be copied to the GPU. This is handled by the GPU buffer manager, which caches input columns on the GPU. If a similar query is run (which is typical for interactive data analysis), the data is already available on the GPU, which significantly accelerates query processing.

The complete query processor is build on a column-oriented, in-memory storage. In case the database does not fit into the main memory, the virtual memory manager of the operating system manages the database buffer. Similar to other main memory optimized DBMSs (e.g., MonetDB), CoGaDB processes a query operator wise. Therefore, CoGaDB executes a complete operator, which consumes its input and materializes its output. Then, the next operator is applied to the previous operators output, until all operators of a query were executed. This processing model allows for efficient caching on the CPU and for coalesced memory accesses on the GPU, which is the key for peak performance. Note that the storage is read only during query processing, because we do not yet support transactions. However, data can be updated offline, when no queries are processed, which fits the typical warehousing process, where data is loaded in a bulk into the database and is then analyzed.

CoGaDB's architecture is summarized in Figure~\ref{fig:cogadb_architecture}.
%
%
%
\endlatexonly

\section sec_coga_interfaces CoGaDB's Query Interfaces

 * CoGaDB has a modular design, meaning it has a layered architecture, where an upper layer is implemented 
 * using the preceding layer.
 * The most advanced way to launch queries in CoGaDB is to use the SQL interface. CoGaDB's SQL parser creates 
 * a logical query plan, consisting of operators from the Logical Operator based API.
 * Using the logical query plan (Logical Operator based API), HyPE creates a physical query plan (which is executable) 
 * by choosing for each operator in the logical query plan a suitable physical operator from the Physical Operator based API.
 * The Physical Operator based API provides a special interface, so HyPE can be used as execution engine. Internally, 
 * the physical operators have to execute a certain operation using the algorithm HyPE selected. This algorithms are 
 * performed in the Function based API, which contains functions that are capable of processing complex operations 
 * (e.g., selections with arbitrary combined filter predicates or groupbys with multiple aggregation functions).
 * The complex functions are implemented using the Internal API, which consists of highly optimized primitives using 
 * libraries such as TBB or Thrust.
 * 
\subsection subsec_sql SQL Interface
 *    - Pass queries in SQL-92 via an interactive shell
 *    - Automatic plan generation, optimization and operator scheduling
 *    - Utility commands not included in SQL
 *    - Recommended API for creating queries in CoGaDB

\subsection subsec_log_op_api Logical Operator based API
 *    - Build queries via API
 *    - Uses HyPE as execution engine, scheduling and query optimization is done automatically (implements the mapping layer for HyPE)
 *    - All available operators can be found in the namespace \ref CoGaDB::query_processing::logical_operator

\subsection subsec_phy_op_api Physical Operator based API
 *    - Build queries via API
 *    - Uses HyPE as execution engine, however, scheduling and query optimization is done manually (bypasses the mapping layer for HyPE)
 *    - All available operators can be found in the namespace \ref CoGaDB::query_processing::physical_operator

\subsection subsec_func_api Function based API
 *    - Build queries via API
 *    - HyPE is not used for execution, so a manual execution of operators (including scheduling and query optimization) is neccessary, for hand-tuned queries (not recommended)
 *    - Calls to CoGaDB's actual database operators
 *    - All available functions can be found in the class \ref CoGaDB::BaseTable

 * 
\subsection subsec_int_api Internal API
 * 
 *    - Internal functions that implement CoGaDB's actual database operators
 *    - Work on single columns either on CPU or GPU
 *    - Each operator has a well defined task it is optimized for (e.g., filter a column, sort a column, join two columns) 
 *      and returns a list of tuple identifiers (TIDs), which are positionlists
 *    - Functions are distributed in different modules, they can be found in the class \ref CoGaDB::ColumnBaseTyped 
 *      and \ref CoGaDB::gpu::GPU_Operators.
 * 
\section Operators
 * We differentiate between to types of operators. Processing operators perfrom computations on the actual data 
 * and can be executed on the CPU or the GPU. Management Operators decompose complex Operations (e.g., filtering 
 * a table according to multiple and arbitrary complex selections or sorting a table after multiple columns).
 * 
\subsection subsec_proc_op Processing Operators
 *    - Selection, 
 *    - Sort
 *    - Join
 *    - Groupby (ColumnAlgebra, AggregationFunctions)
 * 
\subsection subsec_man_op Management Operators 
 *    - Projection (in a column stores it is just skipping some columns while keeping others)
 * 
 * 
*/

/*! \page page_concepts_in_cogadb Concepts

In this section, we describe important concepts and design decisions in CoGaDB. We start with one of the most important building blocks of the query processor, the LookupTables. Then, we discuss the design and capabilities of CoGaDB'S optimizer, divided in the logical and physical optimizer.
 * 
 * \section sec_lookup_tables Lookup Tables
 * 
 * A Lookup Table is a view on one or multiple tables. They are the bridge between the table-based operators and the internal column-based operators. 

Internally, each operator returns the result as a list of TIDs. A LookupTable is basically a list of a pointer to a table, a pointer to a TID list, indicating which tuples of the underlying table belong to the Lookup Table, and a attribute list, specifying which columns of the table are included in the LookupTable. Therefore, LookupTables are a cheap mechanism to store intermediate results. Furthermore, they behave as they were "normal" tables, with the exception that LookupTables cannot be updated. Columns of LookupTables are LookupArrays, which consist of a pointer to a materialized column from a materialized table and a pointer to a TID list. To keep track of which LookupArray indexes a column from which table, we use a helper data structure called LookupColumn. A LookupColumn describes which part of one materialized table is part of a LookupTable, which can be the result of an arbitrary sequence of operators, including binary operators such as joins. 
 * 
 * \section sec_logical_optimization Logical Optimization
 * 
 * CoGaDB implements a simple logical optimizer. It basically implements two of the most basic optimizations: push down selections and resolve cross products by merging them with join conditions to natural joins. To achieve this, CoGaDB has currently four optimizer rules:
 *    -# Break complex selection expressions in conjunctive normal form in a sequence of selections consisting of at most one disjunction
 *    -# Push down the simplefied selections as far as possible. (Either to a SCAN operator, or to a binary operator, where not all conditions in the disjunction fit completely on one subtree, which is typically the case for join conditions.)
 *    -# Now the join conditions were pushed down far enough so they are directly over their respective CROSS_JOIN operators. Therefore, the optimizer removes the join condition, expressed by the selection, and the cross product and replaces them with a semantically equivalent JOIN operator. This process is repeated until all CROSS_JOINS are resolved.
 *    -# In the final step, the optimizer combines succeeding selections (each only one disjunction) to complex selections in conjunctive normal form. This allows for certain optimizations in case two phase physical optimization is used.
 * 
 * 
 * \section sec_physical_optimization Physical Optimization
 * 
 * The core of CoGaDB's physical optimization is the HyPE Library, which is our Hybrid Query Processing Engine. It allocates for each operator in a query plan a processing device and decides on the most suitable algorithm on the selected processing device. Thus, HyPE takes care of the complete physical optimization in CoGaDB.
 * 
 */

/*! \page page_faq FAQ
  \tableofcontents
- What is %CoGaDB?
 - %CoGaDB is a Column-oriented GPU-accelerated DBMS. Its purpose is to be an evaluation platform for researchers who would like to test their own GPU co-processing techniques, query optimization strategies and GPU algorithms.
- Under which License is %CoGaDB distributed?
 - %CoGaDB is released under the GPL v3 License. Therefore, you can download and extend it as you like as long as you obey the terms of the license.
- Can I join the project?
 - Sure, we are always looking for new project members, which help us to extend and improve CoGaDB. You should have basic knowledge in C++ and database implementation techniques. If you are interested in joining the project, contact the development team via <a href="http://wwwiti.cs.uni-magdeburg.de/~bress/">Sebastian\htmlonly Bre&szlig;\endhtmlonly\latexonly Bre{\ss}\endlatexonly</a>.
- I have a technical problem, can I get support?
 - We offer non-commercial support for CoGaDB. In case of questions, suggestions or bug reports, feel free to contact the development team via <a href="http://wwwiti.cs.uni-magdeburg.de/~bress/">Sebastian\htmlonly Bre&szlig;\endhtmlonly\latexonly Bre{\ss}\endlatexonly</a>.


\latexonly
\bibliographystyle{abbrv}
\bibliography{literature}
\endlatexonly

*/
