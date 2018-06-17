//std includes
#include <iostream>
#include <utility>
//boost includes
#include <boost/lexical_cast.hpp>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
//hype includes
#include <query_processing/benchmark.hpp>
//CoGaDB includes
#include <core/table.hpp>
#include <query_processing/query_processor.hpp>
#include <hype.h>

#define HYPE_GENERIC_BENCHMARK_DATATYPE_IS_POINTER
#include <query_processing/generic_operator_benchmark.hpp>
#include <util/hardware_detector.hpp>
#include <boost/algorithm/string.hpp>

#include <time.h>
//#include <util/time_measurement.hpp>
//#include <persistence/storage_manager.hpp>

//Unittests

//#include <gpu/gpu_algorithms.hpp>

//#include <lookup_table/lookup_table.hpp>



//#define USE_ADMISSION_CONTROL
#ifdef USE_ADMISSION_CONTROL
  #include <boost/thread/thread.hpp>
  #include <core/gpu_admission_control.hpp>
#endif



using namespace std;
using namespace CoGaDB;
using namespace query_processing;

            boost::program_options::options_description& getGlobalBenchmarkOptionDescription(){

                static boost::program_options::options_description global_benchmark_option_description("allowed options");
                static bool initialized=false;
                if(!initialized){
                    //add command line parameters
                    global_benchmark_option_description.add_options()
                    ("help", "produce help message")
                    ("number_of_datasets", boost::program_options::value<unsigned int>(), "set the number of data sets for workload")
                    ("number_of_operations", boost::program_options::value<unsigned int>(), "set the number of operations in workload")
                    ("max_dataset_size_in_MB", boost::program_options::value<unsigned int>(), "set the maximal dataset size in MB")
                         //("ready_queue_length", boost::program_options::value<unsigned int>(), "set the queue length of operators that may be concurrently scheduled (clients are blocked on a processing device)")
                    ("scheduling_method", boost::program_options::value<std::string>(), "set the decision model (CPU_ONLY, GPU_ONLY, HYBRID)")
                    ("random_seed", boost::program_options::value<unsigned int>(), "seed to use before for generating datasets and operation workload")
                    ("optimization_criterion", boost::program_options::value<std::string>(), "set the decision models optimization_criterion for all algorithms")
//                    ("statistical_method", boost::program_options::value<std::string>(), "set the decision models statistical_method for all algorithms")
//                    ("recomputation_heuristic", boost::program_options::value<std::string>(), "set the decision models recomputation_heuristic for all algorithms")
                    ;
                    initialized=true;
                }
                return global_benchmark_option_description;
            }

typedef queryprocessing::Generic_Operation_Benchmark<TablePtr>::DeviceSpecifications DeviceSpecifications;
typedef queryprocessing::Generic_Operation_Benchmark<TablePtr>::AlgorithmSpecifications AlgorithmSpecifications;

class Generic_Selection_Benchmark {
public:

    Generic_Selection_Benchmark(const std::string& operation_name,
            const AlgorithmSpecifications& alg_specs,
            const DeviceSpecifications& dev_specs,
            unsigned int number_of_training_iterations,
            unsigned int number_of_parallel_query_sessions)
            : operation_name_(operation_name),
            MAX_DATASET_SIZE_IN_MB_(1), //(10), //MB  //(10*1000*1000)/sizeof(int), //1000000,
            NUMBER_OF_DATASETS_(2), //3, //100,
            NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_(1000), //3, //1000,
            RANDOM_SEED_(0),
            sched_config_(hype::queryprocessing::HYBRID), //(HYBRID), //CPU_ONLY,GPU_ONLY,HYBRID
            stemod_optimization_criterion_("WaitingTimeAwareResponseTime"),
            stemod_statistical_method_("Least Squares 1D"),
            stemod_recomputation_heuristic_("Periodic Recomputation"),
            dev_specs_(dev_specs),
            alg_specs_(alg_specs),
            number_of_training_iterations_(number_of_training_iterations),
            number_of_parallel_query_sessions_(number_of_parallel_query_sessions),
            current_query_id_(0),
            datasets(),
            operator_queries_(),
            query_execution_times_(),
            parallel_user_mutex_(),
            rng_(),
            desc_(getGlobalBenchmarkOptionDescription())
            {
            }

    //Generic_Selection_Benchmark() : hype::queryprocessing::Operation_Benchmark<TablePtr>("SELECTION","CPU_Selection_Algorithm","GPU_Selection_Algorithm"){}

    LogicalQueryPlanPtr generate_logical_operator(TablePtr dataset) {

        //cout << "Create Sort Operation for Table " << dataset->getName() << endl;
        std::string table_name = dataset->getName();

        std::vector<std::string> strs;
        boost::split(strs, table_name, boost::is_any_of("_"));
        unsigned int table_number = boost::lexical_cast<unsigned int>(strs.back());

        int selection_value;
        ValueComparator selection_comparison_value; //0 EQUAL, 1 LESSER, 2 LARGER

        boost::mt19937& rng = this->getRandomNumberGenerator();
        boost::uniform_int<> selection_values(0, 1000);
        boost::uniform_int<> filter_condition(0, 2);

        selection_value = selection_values(rng);
        selection_comparison_value = (ValueComparator) filter_condition(rng); //rand()%3;

        boost::shared_ptr<logical_operator::Logical_Scan> scan(new logical_operator::Logical_Scan(dataset->getName()));

        //boost::shared_ptr<logical_operator::Logical_Selection> selection(new logical_operator::Logical_Selection("values_"+boost::lexical_cast<std::string>(table_number), selection_value, selection_comparison_value, LOOKUP));

        boost::shared_ptr<logical_operator::Logical_Selection> selection(new logical_operator::Logical_Selection("values_"+boost::lexical_cast<std::string>(table_number), selection_value, selection_comparison_value, LOOKUP)); //,hype::DeviceConstraint(GPU_ONLY)));




        selection->setLeft(scan);

        return LogicalQueryPlanPtr(new LogicalQueryPlan(selection));
    }


    //virtual vector<TypedNodePtr> createOperatorQueries() = 0;

    TablePtr generate_dataset(unsigned int size_in_number_of_bytes) {
        static unsigned int dataset_counter = 0;

        cout << "Create Dataset of Size " << size_in_number_of_bytes << " Byte" << endl;

        unsigned int size_in_number_of_elements = size_in_number_of_bytes / sizeof (int);

        TableSchema schema;
        schema.push_back(Attribut(INT, std::string("values_") + boost::lexical_cast<std::string>(dataset_counter)));

        std::string table_name = "Table_";
        table_name += boost::lexical_cast<std::string>(dataset_counter++);
        TablePtr tab1(new Table(table_name, schema));

        boost::mt19937& rng = this->getRandomNumberGenerator();
        boost::uniform_int<> six(0, 1000);

        for (unsigned int i = 0; i < size_in_number_of_elements; ++i) {
            int e = six(rng); //rand();
            //int grouping_value=i/1000; //each group has 1000 keys that needs to be aggregated
            {
                CoGaDB::Tuple t;
                t.push_back(e);
                tab1->insert(t);
            }
        }

        //tab1->print();
        getGlobalTableList().push_back(tab1);


        return tab1;
    }

    std::vector<TablePtr> generate_random_datasets(unsigned int max_size_of_dataset_in_byte, unsigned int number_of_datasets) {
        std::vector<TablePtr> datasets;
        //first, generate dataset of full possible size, then decrease it with each loop according to a value tic, until the last dataset size is only tic
//        unsigned int tic = max_size_of_dataset_in_byte / number_of_datasets;
//        for (unsigned int i = 0; i < number_of_datasets; i++) {
//            TablePtr vec_ptr = this->generate_dataset(max_size_of_dataset_in_byte - i * tic); //(unsigned int) (rand()%max_size_in_number_of_elements) );
//            //assert(vec_ptr!=NULL);
//            datasets.push_back(vec_ptr);
//        }
        //unsigned int tic = max_size_of_dataset_in_byte / number_of_datasets;

        //generate columns of equal size, simulating selections on a denormalized dataware house schema
        for (unsigned int i = 0; i < number_of_datasets; i++) {
            TablePtr vec_ptr = this->generate_dataset(max_size_of_dataset_in_byte); //(unsigned int) (rand()%max_size_in_number_of_elements) );
            //assert(vec_ptr!=NULL);
            datasets.push_back(vec_ptr);
        }
        return datasets;
    }

    int setup(int argc, char* argv[]) {

        //we don't want the OS to swap out our data to disc that's why we lock it
        mlockall(MCL_CURRENT | MCL_FUTURE);

        //	tbb::task_scheduler_init init(8);


        // Declare the supported options.
        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc_), vm);
        boost::program_options::notify(vm);

        if (vm.count("help")) {
            std::cout << desc_ << "\n";
            CoGaDB::exit(EXIT_SUCCESS);
        }

        if (vm.count("number_of_datasets")) {
            std::cout << "Number of Datasets: "
                    << vm["number_of_datasets"].as<unsigned int>() << "\n";
            NUMBER_OF_DATASETS_ = vm["number_of_datasets"].as<unsigned int>();
        } else {
            std::cout << "number_of_datasets was not specified, using default value...\n";
        }

        if (vm.count("number_of_operations")) {
            std::cout << "Number of Operations: "
                    << vm["number_of_operations"].as<unsigned int>() << "\n";
            NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ = vm["number_of_operations"].as<unsigned int>();
        } else {
            std::cout << "number_of_operations was not specified, using default value...\n";
        }

        if (vm.count("max_dataset_size_in_MB")) {
            std::cout << "max_dataset_size_in_MB: "
                    << vm["max_dataset_size_in_MB"].as<unsigned int>() << "MB \n";
            MAX_DATASET_SIZE_IN_MB_ = vm["max_dataset_size_in_MB"].as<unsigned int>(); //*1024*1024)/sizeof(int); //convert value in MB to equivalent number of integer elements
        } else {
            std::cout << "max_dataset_size_in_MB was not specified, using default value...\n";
        }

        if (vm.count("random_seed")) {
            std::cout << "Random Seed: "
                    << vm["random_seed"].as<unsigned int>() << "\n";
            RANDOM_SEED_ = vm["random_seed"].as<unsigned int>();
        } else {
            std::cout << "random_seed was not specified, using default value...\n";
        }


        if (vm.count("scheduling_method")) {
            std::cout << "scheduling_method: "
                    << vm["scheduling_method"].as<std::string>() << "\n";
            std::string scheduling_method = vm["scheduling_method"].as<std::string>();
            if (scheduling_method == "CPU_ONLY") {
                sched_config_ = hype::queryprocessing::CPU_ONLY;
            } else if (scheduling_method == "GPU_ONLY") {
                sched_config_ = hype::queryprocessing::GPU_ONLY;
            } else if (scheduling_method == "HYBRID") {
                sched_config_ = hype::queryprocessing::HYBRID;
            }

        } else {
            std::cout << "scheduling_method was not specified, using default value...\n";
        }

        if (vm.count("optimization_criterion")) {
            std::cout << "optimization_criterion: "
                    << vm["optimization_criterion"].as<std::string>() << "\n";
            stemod_optimization_criterion_ = vm["optimization_criterion"].as<std::string>();

            if (sched_config_ != hype::queryprocessing::HYBRID) {
                std::cout << "Specification of STEMOD Parameter needs hybrid scheduling (scheduling_method=HYBRID)" << std::endl;
                return -1;
            }

        } else {
            std::cout << "optimization_criterion was not specified, using default value...\n";
        }

        if (vm.count("statistical_method")) {
            std::cout << "statistical_method: "
                    << vm["statistical_method"].as<std::string>() << "\n";
            stemod_statistical_method_ = vm["statistical_method"].as<std::string>();
            if (sched_config_ != hype::queryprocessing::HYBRID) {
                std::cout << "Specification of STEMOD Parameter needs hybrid scheduling (scheduling_method=HYBRID)" << std::endl;
                return -1;
            }

        } else {
            std::cout << "statistical_method was not specified, using default value...\n";
        }

        if (vm.count("recomputation_heuristic")) {
            std::cout << "recomputation_heuristic: "
                    << vm["recomputation_heuristic"].as<std::string>() << "\n";
            stemod_recomputation_heuristic_ = vm["recomputation_heuristic"].as<std::string>();
            if (sched_config_ != hype::queryprocessing::HYBRID) {
                std::cout << "Specification of STEMOD Parameter needs hybrid scheduling (scheduling_method=HYBRID)" << std::endl;
                return -1;
            }

        } else {
            std::cout << "recomputation_heuristic was not specified, using default value...\n";
        }


        //"if (vm.count(\"$VAR\")) {
        //    cout << \"$VAR: \"
        // << vm[\"$VAR\"].as<std::string>() << \"\n\";
        //	std::string s=vm[\"$VAR\"].as<std::string>();

        //
        //} else {
        //    cout << \"$VAR was not specified, using default value...\n\";
        //}"

        rng_.seed(RANDOM_SEED_);
        srand(RANDOM_SEED_);

        //
        uint64_t estimated_ram_usage_in_byte = (MAX_DATASET_SIZE_IN_MB_ * 1024 * 1024 * uint64_t(NUMBER_OF_DATASETS_ + 1)) / 2; //MAX_DATASET_SIZE_IN_MB_*NUMBER_OF_DATASETS_

        std::cout << "Generating Data sets..." << std::endl;
        std::cout << "Estimated RAM usage: " << estimated_ram_usage_in_byte / (1024 * 1024) << "MB" << std::endl;
        if ((estimated_ram_usage_in_byte / (1024 * 1024)) > 1024 * 3.7 && hype::util::getArchitecture() == Architecture_32Bit) {
            std::cout << "Warning: Memory for Datasets to generate exceeds 32 bit adress space! (" << estimated_ram_usage_in_byte / (1024 * 1024) << "MB)"
                    << std::endl;
        }
        //generate_random_datasets expects data size in number of integer elements, while MAX_DATASET_SIZE_IN_MB_ specifies data size in Mega Bytes
        datasets = generate_random_datasets((MAX_DATASET_SIZE_IN_MB_ * 1024 * 1024), NUMBER_OF_DATASETS_);

        std::vector<unsigned int> query_indeces(NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_);

        boost::uniform_int<> six(0, NUMBER_OF_DATASETS_ - 1); //choose data sets for sorting equally distributed
        //generate queries
        for (unsigned int i = 0; i < NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_; ++i) {
            query_indeces[i] = six(rng_);
        }
        //std::generate(query_indeces.begin(), query_indeces.end(), Random_Number_Generator(NUMBER_OF_DATASETS_));
        //std::copy(query_indeces.begin(), query_indeces.end(), std::ostream_iterator<unsigned int>(std::cout, "\n"));

        for (unsigned int i = 0; i < NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_; ++i) {

            unsigned int index = query_indeces[i];
            TablePtr dataset = datasets[index];
            LogicalQueryPlanPtr op = generate_logical_operator(dataset);
            assert(op != NULL);
            operator_queries_.push_back(op);
        }

        //std::copy(query_indeces.begin(), query_indeces.end(), std::ostream_iterator<unsigned int>(std::cout, "\n"));

        //setup HyPE
        std::cout << "Setting Optimization Criterion '" << stemod_optimization_criterion_ << "'...";
        if (!hype::Scheduler::instance().setOptimizationCriterion(operation_name_, stemod_optimization_criterion_)) {
            std::cout << "Error: Could not set '" << stemod_optimization_criterion_ << "' as Optimization Criterion!" << std::endl;
            return -1;
        } else std::cout << "Success..." << std::endl;
        //if(!scheduler.setOptimizationCriterion("MERGE","Throughput")) std::cout << "Error" << std::endl;

        //typedef std::vector<DeviceSpecification> DeviceSpecifications dev_specs = CoGaDB::HardwareDetector::instance().getDeviceSpecifications();

//        for (unsigned int i = 0; i < dev_specs_.size(); ++i) {
//            for (unsigned int j = 0; j < alg_specs_.size(); ++j) {
//                //Scheduler::instance().addAlgorithm(alg_specs_[j],dev_specs_[i]);
//                /* set statistical method */
//                if (!hype::Scheduler::instance().setStatisticalMethod(alg_specs_[j].getAlgorithmName(), dev_specs_[i], stemod_statistical_method_)) {
//                    std::cout << "Error setting StatisticalMethod " << stemod_statistical_method_ << " for algorithm: " << alg_specs_[j].getAlgorithmName() << std::endl;
//                    return -1;
//                }
//                /* set recomputation heuristic */
//                if (!hype::Scheduler::instance().setRecomputationHeuristic(alg_specs_[j].getAlgorithmName(), dev_specs_[i], stemod_recomputation_heuristic_)) {
//                    std::cout << "Error setting RecomputationHeuristic " << stemod_statistical_method_ << " for algorithm: " << alg_specs_[j].getAlgorithmName() << std::endl;
//                    return -1;
//                }
//            }
//        }

        //ensure we have enough slots to train all algorithms on each processing device
        assert(hype::core::Runtime_Configuration::instance().getMaximalReadyQueueLength() >= alg_specs_.size() * hype::core::Runtime_Configuration::instance().getTrainingLength());
        //we need at least so many operators in the workload, so we can compelte our training phase
        assert(this->NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ >= alg_specs_.size() * hype::core::Runtime_Configuration::instance().getTrainingLength());

        return 0;
    }

    LogicalQueryPlanPtr getNextQuery() {

        LogicalQueryPlanPtr plan;
        {   //lock mutex, so that each user thread gets its own query
            boost::lock_guard<boost::mutex> lock(this->parallel_user_mutex_);
            if (current_query_id_ < operator_queries_.size()) {
                plan = operator_queries_[current_query_id_];
                //cleanup this plan to avoid resource leak
                operator_queries_[current_query_id_]=LogicalQueryPlanPtr();
                current_query_id_++;
            } else {
                return LogicalQueryPlanPtr();
            }
        }
        return plan;
    }

    void addQueryExecutionTime(double exec_time){
        boost::lock_guard<boost::mutex> lock(this->parallel_user_mutex_);
        query_execution_times_.push_back(exec_time);
//        static unsigned int times_called=0;
//        times_called++;
//        std::cout << "addQueryExecutionTime Times called: " << times_called << std::endl;
        //std::cout << "Query Execution Time: " <<  exec_time << "ns"<< std::endl;
    }

    void query_thread(unsigned int thread_id) {
        LogicalQueryPlanPtr plan = getNextQuery();
        while(plan!=NULL){
            PhysicalQueryPlanPtr phy_plan = plan->runChoppedPlan();
            double exec_time = phy_plan->getExecutionTime();
            addQueryExecutionTime(exec_time);
            plan = getNextQuery();
        }

    }

    void run() {


        Scheduler::instance().setGlobalLoadAdaptionPolicy(No_Recomputation);
        hype_printStatus();
        std::cout << "Starting Benchmark" << std::endl;
        DeviceSpecifications dev_specs = HardwareDetector::instance().getDeviceSpecifications();
        unsigned int num_cpus=0;
        unsigned int num_gpus=0;
        for(unsigned int i=0;i<dev_specs.size();++i){
            if(dev_specs[i].getDeviceType()==hype::CPU){
                num_cpus++;
            }else if(dev_specs[i].getDeviceType()==hype::GPU){
                num_gpus++;
            }
        }




     clock_t start_cpu_time_measurement, end_cpu_time_measurement;
     double cpu_time_used;

     start_cpu_time_measurement = clock();

        Timestamp begin_workload = getTimestamp();

        //first, process queries serially in training phase
        for (unsigned int i = 0; i < number_of_training_iterations_; ++i) {
            LogicalQueryPlanPtr plan = getNextQuery();
            if (plan) {
                plan->runChoppedPlan();
            } else {
                COGADB_FATAL_ERROR("Could not finish training phase: workload to small!", "");
            }
        }
        Timestamp end_training_phase = getTimestamp();

        //after training phase finished, process queries in parallel
        boost::thread_group threads;
        for (unsigned int i = 0; i < number_of_parallel_query_sessions_; i++) {
            threads.add_thread(new boost::thread(boost::bind(&Generic_Selection_Benchmark::query_thread, this, i)));
        }
        threads.join_all();
        Timestamp end_workload = getTimestamp();
        end_cpu_time_measurement = clock();
        cpu_time_used = ((double) (end_cpu_time_measurement - start_cpu_time_measurement)) / CLOCKS_PER_SEC;

        std::cout << "Training Time: " << double(end_training_phase-begin_workload)/(1000*1000) << "ms" << std::endl;
        std::cout << "Workload Time: " << double(end_workload-begin_workload)/(1000*1000) << "ms" << std::endl;
        std::cout << "Used CPU Time: " << cpu_time_used << "s" << std::endl;

        double num_scheduled_gpu_operators=0;
        double num_aborted_gpu_operators=0;
        double wasted_time_by_gpu_operator_abortions=0;
        std::pair<bool, double> num_scheduled_gpu_operators_pair = StatisticsManager::instance().getValue("NUMBER_OF_EXECUTED_GPU_OPERATORS");
        std::pair<bool, double> num_aborted_gpu_operators_pair = StatisticsManager::instance().getValue("NUMBER_OF_ABORTED_GPU_OPERATORS");
        std::pair<bool, double> wasted_time_by_gpu_operator_abortions_pair = StatisticsManager::instance().getValue("TOTAL_LOST_TIME_IN_NS_DUE_TO_ABORTED_GPU_OPERATORS");


        if(num_scheduled_gpu_operators_pair.first){
            num_scheduled_gpu_operators=num_scheduled_gpu_operators_pair.second;
        }

        if(num_aborted_gpu_operators_pair.first){
            num_aborted_gpu_operators=num_aborted_gpu_operators_pair.second;
        }
        if(wasted_time_by_gpu_operator_abortions_pair.first){
            wasted_time_by_gpu_operator_abortions=wasted_time_by_gpu_operator_abortions_pair.second;
        }

	unsigned int total_dataset_size_in_bytes = 0;

	for(unsigned int i=0;i<datasets.size();i++){
		total_dataset_size_in_bytes += datasets[i]->getSizeinBytes(); //*sizeof(ElementType);
        }

//        double sum = std::accumulate(query_execution_times_.begin(),query_execution_times_.end(),double(0));
//        size_t samples = query_execution_times_.size();


        double average_query_execution_time = std::accumulate(query_execution_times_.begin(),query_execution_times_.end(),double(0))/query_execution_times_.size();
        double minimal_query_execution_time = *std::min_element(query_execution_times_.begin(),query_execution_times_.end());
        double maximal_query_execution_time = *std::max_element(query_execution_times_.begin(),query_execution_times_.end());
        double variance_max_likelihood_query_execution_time = 0;
        double variance_bessel_correction_query_execution_time = 0;


        double tmp=0;
        for(unsigned int i=0;i<query_execution_times_.size();++i){
            tmp+=(query_execution_times_[i]-average_query_execution_time)*(query_execution_times_[i]-average_query_execution_time);
        }
        variance_max_likelihood_query_execution_time=tmp/query_execution_times_.size();
        variance_bessel_correction_query_execution_time=tmp/(query_execution_times_.size()-1);

//        std::cout << "SUM: " << sum << " Samples: " << samples << " AVG: " << average_query_execution_time << std::endl;
//        std::cout << "#queries: " << this->operator_queries_.size() << std::endl;

 std::stringstream str_stream;
 str_stream << MAX_DATASET_SIZE_IN_MB_ << "\t"
		<< NUMBER_OF_DATASETS_ << "\t"
		<< NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ << "\t"
		<< sched_config_ << "\t"
		<< total_dataset_size_in_bytes << "\t"
		<< RANDOM_SEED_ << "\t"
		<< stemod_optimization_criterion_ << "\t"
		<< hype::core::Runtime_Configuration::instance().getMaximalReadyQueueLength() << "\t"
		<< hype::core::Runtime_Configuration::instance().getHistoryLength() << "\t"
		<< hype::core::Runtime_Configuration::instance().getRecomputationPeriod() << "\t"
		<< hype::core::Runtime_Configuration::instance().getTrainingLength() << "\t"
		<< hype::core::Runtime_Configuration::instance().getOutlinerThreshold()<< "\t"
		<< hype::core::Runtime_Configuration::instance().getMaximalSlowdownOfNonOptimalAlgorithm() << "\t"
                << number_of_parallel_query_sessions_ << "\t"
                << num_cpus << "\t"
                << num_gpus << "\t"
		<< static_cast<uint64_t>(end_workload-begin_workload) << "\t"
		<< static_cast<uint64_t>(end_training_phase-begin_workload) << "\t"
		<< static_cast<unsigned int>(num_aborted_gpu_operators) << "\t"
 		<< static_cast<unsigned int>(num_scheduled_gpu_operators) << "\t"
 		<< static_cast<uint64_t>(wasted_time_by_gpu_operator_abortions) << "\t"
		<< static_cast<uint64_t>(average_query_execution_time) << "\t"
		<< static_cast<unsigned int>(minimal_query_execution_time) << "\t"
		<< static_cast<unsigned int>(maximal_query_execution_time) << "\t"
		<< static_cast<unsigned int>(variance_max_likelihood_query_execution_time) << "\t"
		<< static_cast<unsigned int>(variance_bessel_correction_query_execution_time);

                //print total execution times (ns)
//                for(unsigned int i=0;i<proc_devs_.size();++i){
//                        str_stream << proc_dev_execution_times[i] << "\t";
//                }
//                //print percentaged execution time w.r.t. execution time spend on all processing devices
//                for(unsigned int i=0;i<proc_devs_.size();++i){
//                       str_stream << percentaged_execution_times[i] << "\t";
//                }
//                //print relative estimation errors
//                for(unsigned int i=0;i<dev_specs_.size();++i){
//                     for(unsigned int j=0;j<alg_specs_.size();++j){
//                          str_stream << relative_errors[i*alg_specs_.size()+j] << "\t";
//                     }
//                }
//		<< total_time_cpu << "\t"
//		<< total_time_gpu << "\t"
//		<< percentaged_execution_time_on_cpu << "\t"
//		<< percentaged_execution_time_on_gpu << "\t"
//		<< relative_error_cpu_parallel_algorithm << "\t"
//		<< relative_error_gpu_algorithm
		//str_stream << std::endl;

                 std::string result_line=str_stream.str();

                 std::stringstream str_stream_header_line;

                 str_stream_header_line << "MAX_DATASET_SIZE_IN_MB_" << "\t"
		<< "NUMBER_OF_DATASETS_" << "\t"
		<< "NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_" << "\t"
		<< "sched_config_" << "\t"
		<< "total_size_of_datasets_in_bytes" << "\t"
		<< "RANDOM_SEED_" << "\t"
		<< "stemod_optimization_criterion_" << "\t"
		<< "stemod_maximal_ready_queue_length" << "\t"
		<< "stemod_history_length" << "\t"
		<< "stemod_recomputation_period" << "\t"
		<< "stemod_length_of_training_phase" << "\t"
		<< "stemod_outliner_threshold_in_percent" << "\t"
		<< "stemod_maximal_slowdown_of_non_optimal_algorithm" << "\t"
                << "#parallel_scan_operations" << "\t"
                << "#CPUs" << "\t"
                << "#GPUs" << "\t" // << "\t";
		<< "workload_execution_time_in_ns" << "\t"
		<< "execution_time_training_only_in_ns" << "\t"
		<< "num_aborted_gpu_operators" << "\t"
 		<< "num_scheduled_gpu_operators" << "\t"
 		<< "wasted_time_by_gpu_operator_abortions_in_ns" << "\t"
		<< "AVG_query_execution_time_ns" << "\t"
		<< "MIN_query_execution_time_ns" << "\t"
		<< "MAX_query_execution_time_ns" << "\t"
		<< "VAR_max_likelihood_query_execution_time_ns" << "\t"
		<< "VAR_bessel_correction_query_execution_time_ns";

                 std::string result_header_line=str_stream_header_line.str();

                 std::cout <<  result_header_line << std::endl;
                 std::cout <<  result_line << std::endl;
                 std::cout << "======================================================================" << std::endl;

        std::fstream file("benchmark_results.log",std::ios_base::out | std::ios_base::app);

	file.seekg(0, std::ios::end); // put the "cursor" at the end of the file
	unsigned int file_length = file.tellg(); // find the position of the cursor

        if (file_length == 0) { //if file empty, write header
            file << result_header_line << std::endl;
        }

        file << result_line << std::endl;
        file.close();

    }

private:
    std::string operation_name_;
    unsigned int MAX_DATASET_SIZE_IN_MB_; //MB  //(10*1000*1000)/sizeof(int); //1000000;
    unsigned int NUMBER_OF_DATASETS_; //3; //100;
    unsigned int NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_; //3; //1000;
    unsigned int RANDOM_SEED_;

    hype::queryprocessing::SchedulingConfiguration sched_config_; //CPU_ONLY,GPU_ONLY,HYBRID

    std::string stemod_optimization_criterion_;
    std::string stemod_statistical_method_;
    std::string stemod_recomputation_heuristic_;

    DeviceSpecifications dev_specs_;
    AlgorithmSpecifications alg_specs_;
    unsigned int current_query_id_;
    unsigned int number_of_training_iterations_;
    unsigned int number_of_parallel_query_sessions_;

    std::vector<TablePtr> datasets;
    std::vector<LogicalQueryPlanPtr> operator_queries_;
    std::vector<double> query_execution_times_;
    boost::mutex parallel_user_mutex_;

    boost::mt19937 rng_; // produces randomness out of thin air
    boost::program_options::options_description desc_; // see pseudo-random number generators

    boost::mt19937& getRandomNumberGenerator() {
        return rng_;
    }
};




unsigned int processing_device_id_ = hype::PD0;

//bool createCPUDevices(DeviceSpecifications& dev_specs, unsigned int number_of_devices) {
//    if (number_of_devices == 0) return false;
//    for (unsigned int i = 0; i < number_of_devices; ++i) {
//        //assert(this->processing_device_id_ < hype::PD_DMA0);
//        dev_specs.push_back(DeviceSpecification((hype::ProcessingDeviceID)processing_device_id_++, hype::CPU, hype::PD_Memory_0)); //one host CPU
//    }
//    return true;
//}
//
////getFreeGPUMemorySizeInByte
//
//bool createGPUDevices(DeviceSpecifications& dev_specs, unsigned int number_of_devices) {
//    //        int memory_id=hype::PD_Memory_1;
//    //        for(unsigned int i=0;i<nDevices;++i){
//    //            assert(processing_device_id_<hype::PD_DMA0);
//    //            dev_specs_.push_back(DeviceSpecification((hype::ProcessingDeviceID)processing_device_id_++,hype::GPU, (hype::ProcessingDeviceMemoryID)memory_id)); //one dedicated GPU
//    //        }
//
//    //we currently do not support multiple real GPUs anyway, so if we
//    //add additional GPUs, we assume they have the same memory id,
//    //which is expecially handy for varying the number of virtual GPUs for one physical GPU
//    for (unsigned int i = 0; i < number_of_devices; ++i) {
//        //assert(processing_device_id_ < hype::PD_DMA0);
//        dev_specs.push_back(DeviceSpecification((hype::ProcessingDeviceID)processing_device_id_++, hype::GPU, hype::PD_Memory_1, &getFreeGPUMemorySizeInByte)); //one dedicated GPU
//    }
//    return true;
//
//}

#ifdef USE_ADMISSION_CONTROL
void workFunction() {
    AdmissionControl::instance().requestMemory(1000 * 1000 * 1000);
    boost::this_thread::sleep(boost::posix_time::seconds(1));
    AdmissionControl::instance().releaseMemory(1000 * 1000 * 1000);
}
#endif

int main(int argc, char* argv[]) {


#ifdef USE_ADMISSION_CONTROL
    std::cout << "Testing admission control" << std::endl;
    boost::thread workThreads[10];
    for(int i=0; i<10; i++) {
        workThreads[i] = boost::thread(&workFunction);
    }
    for(int i=0; i<10; i++) {
        workThreads[i].join();
    }
    std::exit(0);
#endif


    boost::program_options::options_description& desc = getGlobalBenchmarkOptionDescription();
    desc.add_options()
            //("help", "produce help message")
            ("average_operator_selectivity", boost::program_options::value<double>(),
            "average selectivity of the simulated operator has a high impact on the"
            "data transfer overhead of copying results from a co-processor back to the main memory")
            ("parallel_users", boost::program_options::value<unsigned int>(),
            "emulates n parallel selections on a denormalized data warehouse schema, which is realistic even for a single query containing many (n) predicates")
            ("comment", boost::program_options::value<std::string>(), "pass a user comment that will appear in the simulators logfile")
            ;
    //    //backup the new program options (have to be propagated to benchmark, so it
    //    //does not throw an error when parsing simulator options)
    //    boost::program_options::options_description desc_new;
    //    desc_new.add(desc);
    //    //add default benchmark command line options
    //    desc.add(getGlobalBenchmarkOptionDescription());


    AlgorithmSpecifications alg_specs;

    hype::AlgorithmSpecification selection_alg_spec_cpu("CPU_Selection_Algorithm",
            "SELECTION",
            hype::Least_Squares_1D, hype::Periodic);
    hype::AlgorithmSpecification selection_alg_spec_cpu_parallel("CPU_ParallelSelection_Algorithm",
            "SELECTION",
            hype::Least_Squares_1D,
            hype::Periodic);

    hype::AlgorithmSpecification selection_alg_spec_gpu("GPU_Selection_Algorithm",
            "SELECTION",
            hype::Least_Squares_1D,
            hype::Periodic);

    alg_specs.push_back(selection_alg_spec_cpu);
    alg_specs.push_back(selection_alg_spec_cpu_parallel);
    alg_specs.push_back(selection_alg_spec_gpu);

    DeviceSpecifications dev_specs;
    //dev_specs = HardwareDetector::instance().getDeviceSpecifications();

    //    alg_specs.push_back(AlgorithmSpecification("Simulated_Algorithm",
    //                                                simulated_operator_name,
    //                                                hype::Least_Squares_1D,
    //                                                hype::Periodic,
    //                                                hype::ResponseTime)
    //                       );

    double average_operator_selectivity_value = 1.0; //steers the result sizes, which have to be
    //transferred back from Co-Processor to CPU
    unsigned int number_of_parallel_scan_operations=10;

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);
    vector<string> tokens;
    std::string user_comment;
    if (vm.count("average_operator_selectivity")) {
        average_operator_selectivity_value = vm["average_operator_selectivity"].as<double>();
        assert(average_operator_selectivity_value >= 0 && average_operator_selectivity_value <= 1);
    }
    std::cout << "Average Operator Selectivity: " << average_operator_selectivity_value << std::endl;

    if (vm.count("parallel_users")) {
        number_of_parallel_scan_operations = vm["parallel_users"].as<unsigned int>();

    }
    std::cout << "Number of parallel scans: " << number_of_parallel_scan_operations << std::endl;


    if (vm.count("comment")) {
        user_comment = vm["comment"].as<std::string>();
    }
    std::cout << "User Comment: " << user_comment << std::endl;

    dev_specs = HardwareDetector::instance().getDeviceSpecifications();
    unsigned int num_cpus=0;
    unsigned int num_gpus=0;
    for(unsigned int i=0;i<dev_specs.size();++i){
        if(dev_specs[i].getDeviceType()==hype::CPU){
            num_cpus++;
        }else if(dev_specs[i].getDeviceType()==hype::GPU){
            num_gpus++;
        }
    }

    Generic_Selection_Benchmark s("SELECTION", alg_specs, dev_specs, 3*(num_cpus+num_gpus),number_of_parallel_scan_operations);

    s.setup(argc, argv);

    s.run();

//    dev_specs = HardwareDetector::instance().getDeviceSpecifications();
//    unsigned int num_cpus=0;
//    unsigned int num_gpus=0;
//    for(unsigned int i=0;i<dev_specs.size();++i){
//        if(dev_specs[i].getDeviceType()==hype::CPU){
//            num_cpus++;
//        }else if(dev_specs[i].getDeviceType()==hype::GPU){
//            num_gpus++;
//        }
//    }

    //write log file infos here


    return 0;
}
