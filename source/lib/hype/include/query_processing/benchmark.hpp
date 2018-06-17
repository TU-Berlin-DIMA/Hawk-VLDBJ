
#pragma once

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include <stdint.h>
#include <sys/mman.h>

#include <config/configuration.hpp>
#include <hype.hpp>
#include <query_processing/logical_query_plan.hpp>
#include <query_processing/node.hpp>
#include <query_processing/operator.hpp>
#include <query_processing/processing_device.hpp>
#include <util/architecture.hpp>
#include <util/hardware_detector.hpp>

#include <tbb/parallel_sort.h>
#include <tbb/task_scheduler_init.h>

#include <boost/chrono.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/thread.hpp>

//#ifndef ENABLE_GPU_ACCELERATION
//        #define ENABLE_GPU_ACCELERATION
//#endif

namespace hype {

  namespace queryprocessing {

    template <typename Type, typename DataSetType = Type>
    class Operation_Benchmark {
     public:
      struct Random_Number_Generator {
        Random_Number_Generator(unsigned int max_value_size)
            : max_value_size_(max_value_size) {}

        unsigned int operator()() {
          return (unsigned int)rand() % max_value_size_;
        }

       private:
        unsigned int max_value_size_;
      };

      typedef Type type;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedOperatorPtr
          TypedOperatorPtr;
      typedef
          typename OperatorMapper_Helper_Template<Type>::Physical_Operator_Map
              Physical_Operator_Map;
      typedef typename OperatorMapper_Helper_Template<
          Type>::Physical_Operator_Map_Ptr Physical_Operator_Map_Ptr;
      typedef typename OperatorMapper_Helper_Template<
          Type>::Create_Typed_Operator_Function Create_Typed_Operator_Function;
      typedef typename OperatorMapper_Helper_Template<Type>::TypedNodePtr
          TypedNodePtr;

      /*
      typedef OperatorMapper_Helper_Template<Type>::TypedOperatorPtr
      TypedOperatorPtr;
      typedef OperatorMapper_Helper_Template<Type>::Physical_Operator_Map
      Physical_Operator_Map;
      typedef OperatorMapper_Helper_Template<Type>::Physical_Operator_Map_Ptr
      Physical_Operator_Map_Ptr;
      typedef
      OperatorMapper_Helper_Template<Type>::Create_Typed_Operator_Function
      Create_Typed_Operator_Function;
      typedef OperatorMapper_Helper_Template<Type>::TypedNodePtr TypedNodePtr;
      */

      // typedef int ElementType;
      // typedef vector<ElementType> Vec;
      // typedef boost::shared_ptr<Vec> VecPtr;

      Operation_Benchmark(const std::string& operation_name,
                          const std::string& cpu_algorithm_name,
                          const std::string& gpu_algorithm_name)
          : operation_name_(operation_name),
            cpu_algorithm_name_(cpu_algorithm_name),
            gpu_algorithm_name_(gpu_algorithm_name),
            MAX_DATASET_SIZE_IN_MB_(
                1),  //(10), //MB  //(10*1000*1000)/sizeof(int), //1000000,
            NUMBER_OF_DATASETS_(10),                     // 3, //100,
            NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_(30),  // 3, //1000,
            RANDOM_SEED_(0),
            sched_config_(CPU_ONLY),  //(HYBRID), //CPU_ONLY,GPU_ONLY,HYBRID
            stemod_optimization_criterion_("Response Time"),
            stemod_statistical_method_("Least Squares 1D"),
            stemod_recomputation_heuristic_("Periodic Recomputation"),
            cpu_dev_spec_(
                DeviceSpecification(hype::PD0, hype::CPU, hype::PD_Memory_0)),
            gpu_dev_spec_(
                DeviceSpecification(hype::PD1, hype::GPU, hype::PD_Memory_1)),
            cpu(),
            gpu(),
            datasets(),
            operator_queries_(),
            rng_()  //,
      // operator_map_()
      {
        //              if(cpu==NULL ||  gpu==NULL){
        //                  std::cout << "HyPE: FATAL ERROR:
        //                  Operation_Benchmark: At least one Processing Device
        //                  not found! "
        //                       << "This class assumes you setup the algorithms
        //                       and processing devices already, and that "
        //                       << "you have one CPU (DeviceSpecification
        //                       (hype::PD0,hype::CPU,hype::PD_Memory_0)) "
        //                       << "and one GPU
        //                       (DeviceSpecification(hype::PD1,hype::GPU,hype::PD_Memory_1))"
        //                       << std::endl;
        //                  std::exit(-1);
        //              }
        //            cpu=hype::queryprocessing::getProcessingDevice(DeviceSpecification
        //            (hype::PD0,hype::CPU,hype::PD_Memory_0));
        //            gpu=hype::queryprocessing::getProcessingDevice(DeviceSpecification
        //            (hype::PD1,hype::GPU,hype::PD_Memory_1));
        /*
        if(!setup(argc, argv)){
                std::cout << "Benchmark Setup Failed!" << std::endl;
                std::exit(-1);
        }*/
      }

      // RandomNumberGenerator random_number_generator_;
      std::string operation_name_;
      std::string cpu_algorithm_name_;
      std::string gpu_algorithm_name_;

      unsigned int MAX_DATASET_SIZE_IN_MB_;  // MB
                                             // //(10*1000*1000)/sizeof(int);
                                             // //1000000;
      unsigned int NUMBER_OF_DATASETS_;      // 3; //100;
      unsigned int NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_;  // 3; //1000;
      unsigned int RANDOM_SEED_;

      SchedulingConfiguration sched_config_;  // CPU_ONLY,GPU_ONLY,HYBRID

      std::string stemod_optimization_criterion_;
      std::string stemod_statistical_method_;
      std::string stemod_recomputation_heuristic_;

      DeviceSpecification cpu_dev_spec_;
      DeviceSpecification gpu_dev_spec_;
      hype::queryprocessing::ProcessingDevicePtr cpu;
      hype::queryprocessing::ProcessingDevicePtr gpu;

      std::vector<DataSetType> datasets;
      std::vector<TypedNodePtr> operator_queries_;

      boost::mt19937 rng_;  // produces randomness out of thin air
                            // see pseudo-random number generators

      boost::mt19937& getRandomNumberGenerator() { return rng_; }

      // Physical_Operator_Map_Ptr operator_map_;

      uint64_t getTimestamp() {
        using namespace boost::chrono;

        high_resolution_clock::time_point tp = high_resolution_clock::now();
        nanoseconds dur = tp.time_since_epoch();

        return (uint64_t)dur.count();
      }

      virtual TypedNodePtr generate_logical_operator(DataSetType dataset) = 0;

      // virtual vector<TypedNodePtr> createOperatorQueries() = 0;

      virtual DataSetType generate_dataset(
          unsigned int size_in_number_of_bytes) = 0;

      // virtual destructor
      virtual ~Operation_Benchmark() {}

      // Type generate_dataset(unsigned int size_in_number_of_elements){
      //	VecPtr data(new Vec());
      //	for(unsigned int i=0;i<size_in_number_of_elements;i++){
      //		ElementType e = (ElementType) rand();
      //		data->push_back(e);
      //	}
      //	assert(data!=NULL);
      //	//std::cout << "created new data set: " << data.get() << " of
      // size: " << data->size() << std::endl;
      //	return data;
      //}

      std::vector<DataSetType> generate_random_datasets(
          unsigned int max_size_of_dataset_in_byte,
          unsigned int number_of_datasets) {
        std::vector<DataSetType> datasets;
        // first, generate dataset of full possible size, then decrease it with
        // each loop according to a value tic, until the last dataset size is
        // only tic
        unsigned int tic = max_size_of_dataset_in_byte / number_of_datasets;
        for (unsigned int i = 0; i < number_of_datasets; i++) {
          DataSetType vec_ptr = this->generate_dataset(
              max_size_of_dataset_in_byte - i * tic);  //(unsigned int)
          //(rand()%max_size_in_number_of_elements) );
          assert(vec_ptr != NULL);
          datasets.push_back(vec_ptr);
        }
        return datasets;
      }

      int setup(int argc, char* argv[]) {
        // we don't want the OS to swap out our data to disc that's why we lock
        // it
        int ret = 0;
        // ret=mlockall(MCL_CURRENT|MCL_FUTURE);
        if (ret != 0) {
          std::cout
              << "FATAL ERROR! Could not lock main memory! Try run as root."
              << std::endl;
          exit(-1);
        }
        //	tbb::task_scheduler_init init(8);

        // tbb::task_scheduler_init (2);
        //	tbb::task_scheduler_init init(1);
        //

        //	cout << "TBB use " <<
        // tbb::task_scheduler_init::default_num_threads() << " number of
        // threads
        // as a default" << std::endl;
        //	unsigned int MAX_DATASET_SIZE_IN_MB_=10; //MB
        ////(10*1000*1000)/sizeof(int); //1000000;
        //	unsigned int NUMBER_OF_DATASETS_=10; //3; //100;
        //	unsigned int NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_=100; //3;
        ////1000;
        //	unsigned int RANDOM_SEED_=0;
        //	//unsigned int READY_QUEUE_LENGTH=100;

        //	SchedulingConfiguration sched_config_=HYBRID;
        ////CPU_ONLY,GPU_ONLY,HYBRID
        //	//SchedulingConfiguration sched_config_=GPU_ONLY;
        //	//SchedulingConfiguration sched_config_=CPU_ONLY;

        //	std::string stemod_optimization_criterion_="Response Time";
        //	std::string stemod_statistical_method_="Least Squares 1D";
        //	std::string stemod_recomputation_heuristic_="Periodic
        // Recomputation";

        // Declare the supported options.
        boost::program_options::options_description desc("Allowed options");
        desc.add_options()("help", "produce help message")(
            "number_of_datasets", boost::program_options::value<unsigned int>(),
            "set the number of data sets for workload")(
            "number_of_operations",
            boost::program_options::value<unsigned int>(),
            "set the number of operations in workload")(
            "max_dataset_size_in_MB",
            boost::program_options::value<unsigned int>(),
            "set the maximal dataset size in MB")
            //("ready_queue_length", boost::program_options::value<unsigned
            // int>(), "set the queue length of operators that may be
            // concurrently scheduled (clients are blocked on a processing
            // device)")
            ("scheduling_method", boost::program_options::value<std::string>(),
             "set the decision model (CPU_ONLY, GPU_ONLY, HYBRID)")(
                "random_seed", boost::program_options::value<unsigned int>(),
                "seed to use before for generating datasets and operation "
                "workload")("optimization_criterion",
                            boost::program_options::value<std::string>(),
                            "set the decision models optimization_criterion "
                            "for all algorithms")(
                "statistical_method",
                boost::program_options::value<std::string>(),
                "set the decision models statistical_method for all "
                "algorithms")("recomputation_heuristic",
                              boost::program_options::value<std::string>(),
                              "set the decision models recomputation_heuristic "
                              "for all algorithms");

        boost::program_options::variables_map vm;
        boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        if (vm.count("help")) {
          std::cout << desc << "\n";
          return 1;
        }

        if (vm.count("number_of_datasets")) {
          std::cout << "Number of Datasets: "
                    << vm["number_of_datasets"].as<unsigned int>() << "\n";
          NUMBER_OF_DATASETS_ = vm["number_of_datasets"].as<unsigned int>();
        } else {
          std::cout << "number_of_datasets was not specified, using default "
                       "value...\n";
        }

        if (vm.count("number_of_operations")) {
          std::cout << "Number of Operations: "
                    << vm["number_of_operations"].as<unsigned int>() << "\n";
          NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ =
              vm["number_of_operations"].as<unsigned int>();
        } else {
          std::cout << "number_of_operations was not specified, using default "
                       "value...\n";
        }

        if (vm.count("max_dataset_size_in_MB")) {
          std::cout << "max_dataset_size_in_MB: "
                    << vm["max_dataset_size_in_MB"].as<unsigned int>()
                    << "MB \n";
          MAX_DATASET_SIZE_IN_MB_ =
              vm["max_dataset_size_in_MB"]
                  .as<unsigned int>();  //*1024*1024)/sizeof(int); //convert
          // value in MB to equivalent number of
          // integer elements
        } else {
          std::cout << "max_dataset_size_in_MB was not specified, using "
                       "default value...\n";
        }

        if (vm.count("random_seed")) {
          std::cout << "Random Seed: " << vm["random_seed"].as<unsigned int>()
                    << "\n";
          RANDOM_SEED_ = vm["random_seed"].as<unsigned int>();
        } else {
          std::cout
              << "random_seed was not specified, using default value...\n";
        }

        if (vm.count("scheduling_method")) {
          std::cout << "scheduling_method: "
                    << vm["scheduling_method"].as<std::string>() << "\n";
          std::string scheduling_method =
              vm["scheduling_method"].as<std::string>();
          if (scheduling_method == "CPU_ONLY") {
            sched_config_ = CPU_ONLY;
          } else if (scheduling_method == "GPU_ONLY") {
            sched_config_ = GPU_ONLY;
          } else if (scheduling_method == "HYBRID") {
            sched_config_ = HYBRID;
          }

        } else {
          std::cout << "scheduling_method was not specified, using default "
                       "value...\n";
        }

        if (vm.count("optimization_criterion")) {
          std::cout << "optimization_criterion: "
                    << vm["optimization_criterion"].as<std::string>() << "\n";
          stemod_optimization_criterion_ =
              vm["optimization_criterion"].as<std::string>();

          if (sched_config_ != HYBRID) {
            std::cout << "Specification of STEMOD Parameter needs hybrid "
                         "scheduling (scheduling_method=HYBRID)"
                      << std::endl;
            return -1;
          }

        } else {
          std::cout << "optimization_criterion was not specified, using "
                       "default value...\n";
        }

        if (vm.count("statistical_method")) {
          std::cout << "statistical_method: "
                    << vm["statistical_method"].as<std::string>() << "\n";
          stemod_statistical_method_ =
              vm["statistical_method"].as<std::string>();
          if (sched_config_ != HYBRID) {
            std::cout << "Specification of STEMOD Parameter needs hybrid "
                         "scheduling (scheduling_method=HYBRID)"
                      << std::endl;
            return -1;
          }

        } else {
          std::cout << "statistical_method was not specified, using default "
                       "value...\n";
        }

        if (vm.count("recomputation_heuristic")) {
          std::cout << "recomputation_heuristic: "
                    << vm["recomputation_heuristic"].as<std::string>() << "\n";
          stemod_recomputation_heuristic_ =
              vm["recomputation_heuristic"].as<std::string>();
          if (sched_config_ != HYBRID) {
            std::cout << "Specification of STEMOD Parameter needs hybrid "
                         "scheduling (scheduling_method=HYBRID)"
                      << std::endl;
            return -1;
          }

        } else {
          std::cout << "recomputation_heuristic was not specified, using "
                       "default value...\n";
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
        uint64_t estimated_ram_usage_in_byte =
            (MAX_DATASET_SIZE_IN_MB_ * 1024 * 1024 *
             uint64_t(NUMBER_OF_DATASETS_ + 1)) /
            2;  // MAX_DATASET_SIZE_IN_MB_*NUMBER_OF_DATASETS_
                /*	unsigned int
                   tic=MAX_DATASET_SIZE_IN_MB_*1024*1024/NUMBER_OF_DATASETS_;
                        std::cout << "tic size: " << tic << std::endl;
                                for(unsigned int i=0;i<NUMBER_OF_DATASETS_;i++){
                                        std::cout << "Size: " <<
                   MAX_DATASET_SIZE_IN_MB_*1024*1024-i*tic << std::endl;
                                        estimated_ram_usage_in_byte+=MAX_DATASET_SIZE_IN_MB_*1024*1024-i*tic;
                                }*/

        std::cout << "Generating Data sets..." << std::endl;
        std::cout << "Estimated RAM usage: "
                  << estimated_ram_usage_in_byte / (1024 * 1024) << "MB"
                  << std::endl;
        if ((estimated_ram_usage_in_byte / (1024 * 1024)) > 1024 * 3.7 &&
            util::getArchitecture() == Architecture_32Bit) {
          std::cout << "Memory for Datasets to generate exceeds 32 bit adress "
                       "space! ("
                    << estimated_ram_usage_in_byte / (1024 * 1024) << "MB)"
                    << std::endl
                    << "Exiting..." << std::endl;
          std::exit(-1);
        }
        // generate_random_datasets expects data size in number of integer
        // elements, while MAX_DATASET_SIZE_IN_MB_ specifies data size in Mega
        // Bytes
        datasets = generate_random_datasets(
            (MAX_DATASET_SIZE_IN_MB_ * 1024 * 1024), NUMBER_OF_DATASETS_);

        std::vector<unsigned int> query_indeces(
            NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_);

        boost::uniform_int<> six(
            0, NUMBER_OF_DATASETS_ -
                   1);  // choose data sets for sorting equally distributed
        // generate queries
        for (unsigned int i = 0; i < NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_;
             ++i) {
          query_indeces[i] = six(rng_);
        }
        // std::generate(query_indeces.begin(), query_indeces.end(),
        // Random_Number_Generator(NUMBER_OF_DATASETS_));
        // std::copy(query_indeces.begin(), query_indeces.end(),
        // std::ostream_iterator<unsigned int>(std::cout, "\n"));

        for (unsigned int i = 0; i < NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_;
             ++i) {
          unsigned int index = query_indeces[i];
          DataSetType dataset = datasets[index];
          TypedNodePtr op = generate_logical_operator(dataset);
          assert(op != NULL);
          operator_queries_.push_back(op);
        }

        // std::copy(query_indeces.begin(), query_indeces.end(),
        // std::ostream_iterator<unsigned int>(std::cout, "\n"));

        // setup STEMOD
        // stemod::Scheduler::instance().addAlgorithm(operation_name_,"CPU_Algorithm_serial","Least
        // Squares 1D","Periodic Recomputation");
        // stemod::Scheduler::instance().addAlgorithm(operation_name_,cpu_algorithm_name_,
        // stemod::CPU, "Least Squares 1D", "Periodic Recomputation");
        // stemod::Scheduler::instance().addAlgorithm(operation_name_,gpu_algorithm_name_,
        // stemod::GPU, "Least Squares 1D", "Periodic Recomputation");

        std::cout << "Setting Optimization Criterion '"
                  << stemod_optimization_criterion_ << "'...";
        if (!hype::Scheduler::instance().setOptimizationCriterion(
                operation_name_, stemod_optimization_criterion_)) {
          std::cout << "Error: Could not set '"
                    << stemod_optimization_criterion_
                    << "' as Optimization Criterion!" << std::endl;
          return -1;
        } else
          std::cout << "Success..." << std::endl;
        // if(!scheduler.setOptimizationCriterion("MERGE","Throughput"))
        // std::cout << "Error" << std::endl;

        const CoGaDB::DeviceSpecifications& dev_specs =
            CoGaDB::HardwareDetector::instance().getDeviceSpecifications();
        /* set statistical method */
        for (unsigned int i = 0; i < dev_specs.size(); ++i) {
          if (dev_specs[i].getDeviceType() == hype::CPU) {
            if (!hype::Scheduler::instance().setStatisticalMethod(
                    cpu_algorithm_name_, dev_specs[i],
                    stemod_statistical_method_)) {
              std::cout << "Error" << std::endl;
              return -1;
            }
          } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
            if (!hype::Scheduler::instance().setStatisticalMethod(
                    gpu_algorithm_name_, dev_specs[i],
                    stemod_statistical_method_)) {
              std::cout << "Error" << std::endl;
              return -1;
            }
#endif
          }
        }

        /* set recomputation heuristic */
        for (unsigned int i = 0; i < dev_specs.size(); ++i) {
          if (dev_specs[i].getDeviceType() == hype::CPU) {
            if (!hype::Scheduler::instance().setRecomputationHeuristic(
                    cpu_algorithm_name_, dev_specs[i],
                    stemod_recomputation_heuristic_)) {
              std::cout << "Error" << std::endl;
              return -1;
            }
          } else if (dev_specs[i].getDeviceType() == hype::GPU) {
#ifdef ENABLE_GPU_ACCELERATION
            if (!hype::Scheduler::instance().setRecomputationHeuristic(
                    gpu_algorithm_name_, dev_specs[i],
                    stemod_recomputation_heuristic_)) {
              std::cout << "Error" << std::endl;
              return -1;
            }
#endif
          }
        }

        //	if(!hype::Scheduler::instance().setStatisticalMethod(cpu_algorithm_name_,stemod_statistical_method_)){
        //		std::cout << "Error" << std::endl; return -1;
        //	} else std::cout << "Success..." << std::endl;
        //	if(!hype::Scheduler::instance().setStatisticalMethod(gpu_algorithm_name_,stemod_statistical_method_)){
        //		std::cout << "Error" << std::endl; return -1;
        //	} else std::cout << "Success..." << std::endl;
        //
        //	if(!hype::Scheduler::instance().setRecomputationHeuristic(cpu_algorithm_name_,stemod_recomputation_heuristic_)){
        //		std::cout << "Error" << std::endl; return -1;
        //	}	else std::cout << "Success..." << std::endl;
        //	if(!hype::Scheduler::instance().setRecomputationHeuristic(gpu_algorithm_name_,stemod_recomputation_heuristic_)){
        //		std::cout << "Error" << std::endl; return -1;
        //	} else std::cout << "Success..." << std::endl;

        cpu = hype::queryprocessing::getProcessingDevice(cpu_dev_spec_);
        gpu = hype::queryprocessing::getProcessingDevice(gpu_dev_spec_);

        assert(cpu != NULL);
        assert(gpu != NULL);

        cpu->start();
        gpu->start();

        return 0;
      }

      int run() {
        // boost::this_thread::sleep( boost::posix_time::seconds(30) );

        std::cout << "Starting Benchmark..." << std::endl;

        uint64_t begin_benchmark_timestamp = getTimestamp();
        uint64_t end_training_timestamp = 0;

        for (unsigned int i = 0; i < NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_;
             i++) {
          /*
          unsigned int index = query_indeces[i];
          VecPtr dataset = datasets[index];

          assert(NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_==query_indeces.size());
          assert(index<NUMBER_OF_DATASETS_);
          //NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_);

          stemod::Tuple t;
          t.push_back(dataset->size());
          //stemod::SchedulingDecision
          sched_dec_local("",stemod::core::EstimatedTime(0),t);
          */

          TypedNodePtr current_operator = operator_queries_[i];
          TypedNodePtr scan_left =
              boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                  current_operator->getLeft());
          TypedNodePtr scan_right =
              boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                  current_operator->getRight());

          // std::cout << "RUN: " << i << "/" <<
          // NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ << std::endl;

          if (sched_config_ == HYBRID) {  // CPU_ONLY,GPU_ONLY,HYBRID)
            // cout << "scheduling operator " << i << std::endl;
            const unsigned int number_of_training_operations =
                (hype::core::Runtime_Configuration::instance()
                     .getTrainingLength() *
                 2) +
                1;  //*number of algortihms per operation (2)
            if (number_of_training_operations == i) {
              if (!hype::core::quiet)
                std::cout << "waiting for training to complete" << std::endl;
              // wait until training operations finished
              while (!cpu->isIdle() || !gpu->isIdle()) {
                // std::cout << "stat: cpu " << !cpu->isIdle() << " gpu " <<
                // !gpu->isIdle() << std::endl;
                boost::this_thread::sleep(boost::posix_time::microseconds(20));
              }
              end_training_timestamp = getTimestamp();
              // cout << "stat: cpu " << !cpu->isIdle() << " gpu " <<
              // !gpu->isIdle() << std::endl;
              if (!hype::core::quiet)
                std::cout << "training completed! Time: "
                          << end_training_timestamp - begin_benchmark_timestamp
                          << "ns ("
                          << double(end_training_timestamp -
                                    begin_benchmark_timestamp) /
                                 (1000 * 1000 * 1000)
                          << "s)" << std::endl;
            }

            TypedOperatorPtr phy_left_scan;
            TypedOperatorPtr phy_right_scan;

            if (scan_left)
              phy_left_scan = scan_left->getOptimalOperator(TypedOperatorPtr(),
                                                            TypedOperatorPtr());
            if (scan_right)
              phy_right_scan = scan_right->getOptimalOperator(
                  TypedOperatorPtr(), TypedOperatorPtr());

            TypedOperatorPtr phy_op = current_operator->getOptimalOperator(
                phy_left_scan, phy_right_scan);

            if (phy_op->getDeviceSpecification().getDeviceType() == CPU) {
              cpu->addOperator(phy_op);
            } else if (phy_op->getDeviceSpecification().getDeviceType() ==
                       GPU) {
              gpu->addOperator(phy_op);
            }

          } else if (sched_config_ == CPU_ONLY) {
            // CPU_Sort_Parallel(dataset);

            TypedOperatorPtr phy_left_scan;
            TypedOperatorPtr phy_right_scan;

            if (scan_left)
              phy_left_scan = scan_left->getOptimalOperator(TypedOperatorPtr(),
                                                            TypedOperatorPtr());
            if (scan_right)
              phy_right_scan = scan_right->getOptimalOperator(
                  TypedOperatorPtr(), TypedOperatorPtr());

            TypedOperatorPtr phy_op = current_operator->getOptimalOperator(
                phy_left_scan, phy_right_scan, hype::CPU_ONLY);

            assert(phy_op->getDeviceSpecification().getDeviceType() == CPU);
            cpu->addOperator(phy_op);
            /*
            Physical_Operator_Map_Ptr operator_map_=
            current_operator->getPhysical_Operator_Map();

            Create_Typed_Operator_Function
            f=(*operator_map_)[cpu_algorithm_name_];

            f(current_operator,);
            */

            // Physical_Operator_Map_Ptr
            // map=current_operator->getPhysical_Operator_Map();
            // TypedOperatorPtr phy_op=map[cpu_algorithm_name_];
            // cpu->addOperator(phy_op);

            // std::cout << "Assigning Operator to CPU... " << std::endl;
            // cpu->addOperator( boost::shared_ptr<CPU_Parallel_Sort_Operator>(
            // new CPU_Parallel_Sort_Operator(sched_dec_local, dataset) ) );

          } else if (sched_config_ == GPU_ONLY) {
            // GPU_Sort(dataset);

            TypedOperatorPtr phy_left_scan;
            TypedOperatorPtr phy_right_scan;

            if (scan_left)
              phy_left_scan = scan_left->getOptimalOperator(TypedOperatorPtr(),
                                                            TypedOperatorPtr());
            if (scan_right)
              phy_right_scan = scan_right->getOptimalOperator(
                  TypedOperatorPtr(), TypedOperatorPtr());

            TypedOperatorPtr phy_op = current_operator->getOptimalOperator(
                phy_left_scan, phy_right_scan, hype::GPU_ONLY);

            assert(phy_op->getDeviceSpecification().getDeviceType() == GPU);
            gpu->addOperator(phy_op);

            // Physical_Operator_Map_Ptr
            // map=current_operator->getPhysical_Operator_Map();
            // TypedOperatorPtr phy_op=map[gpu_algorithm_name_];
            // gpu->addOperator(phy_op);

            // std::cout << "Assigning Operator to GPU... " << std::endl;
            // gpu->addOperator( boost::shared_ptr<GPU_Sort_Operator>( new
            // GPU_Sort_Operator(sched_dec_local, dataset) ) );
          }

          /*
          stemod::SchedulingDecision sched_dec =
stemod::Scheduler::instance().getOptimalAlgorithmName(operation_name_,t);

  if(sched_dec.getNameofChoosenAlgorithm()=="CPU_Algorithm_serial"){
          cpu->addOperator( boost::shared_ptr<CPU_Serial_Sort_Operator>( new
CPU_Serial_Sort_Operator(sched_dec, dataset) ) );
//			stemod::AlgorithmMeasurement alg_measure(sched_dec);
//			CPU_Sort(dataset);
//			alg_measure.afterAlgorithmExecution();
  }else if(sched_dec.getNameofChoosenAlgorithm()==cpu_algorithm_name_){
          cpu->addOperator( boost::shared_ptr<CPU_Parallel_Sort_Operator>( new
CPU_Parallel_Sort_Operator(sched_dec, dataset) ) );
//			stemod::AlgorithmMeasurement alg_measure(sched_dec);
//			CPU_Sort_Parallel(dataset);
//			alg_measure.afterAlgorithmExecution();
  }else if(sched_dec.getNameofChoosenAlgorithm()==gpu_algorithm_name_){
          gpu->addOperator( boost::shared_ptr<GPU_Sort_Operator>( new
GPU_Sort_Operator(sched_dec, dataset) ) );
//			stemod::AlgorithmMeasurement alg_measure(sched_dec);
//			GPU_Sort(dataset);
//			alg_measure.afterAlgorithmExecution();
  }

  }else if(sched_config_==CPU_ONLY){
          CPU_Sort_Parallel(dataset);
          //std::cout << "Assigning Operator to CPU... " << std::endl;
          //cpu->addOperator( boost::shared_ptr<CPU_Parallel_Sort_Operator>( new
CPU_Parallel_Sort_Operator(sched_dec_local, dataset) ) );
  }else if(sched_config_==GPU_ONLY){
          GPU_Sort(dataset);
          //std::cout << "Assigning Operator to GPU... " << std::endl;
          //gpu->addOperator( boost::shared_ptr<GPU_Sort_Operator>( new
GPU_Sort_Operator(sched_dec_local, dataset) ) );
  }*/
        }

        //	boost::this_thread::sleep( boost::posix_time::seconds(3) );

        //	cpu->stop();
        //	gpu->stop();

        while (!cpu->isIdle() || !gpu->isIdle()) {
          boost::this_thread::sleep(boost::posix_time::microseconds(20));
        }
        uint64_t end_benchmark_timestamp = getTimestamp();
        std::cout << "stat: cpu " << !cpu->isIdle() << " gpu " << !gpu->isIdle()
                  << std::endl;
        std::cout << "[Main Thread] Processing Devices finished..."
                  << std::endl;

        cpu->stop();
        gpu->stop();

        // if one of the following assertiosn are not fulfilled, then abort,
        // because results are rubbish
        assert(end_benchmark_timestamp >= begin_benchmark_timestamp);
        double time_for_training_phase = 0;
        double relative_error_cpu_parallel_algorithm = 0;
        double relative_error_gpu_algorithm = 0;

        if (sched_config_ == HYBRID) {  // a training phase only exists when the
                                        // decision model is used
          assert(end_training_timestamp >= begin_benchmark_timestamp);
          assert(end_benchmark_timestamp >= end_training_timestamp);
          time_for_training_phase =
              end_training_timestamp - begin_benchmark_timestamp;
          relative_error_cpu_parallel_algorithm =
              hype::Report::instance().getRelativeEstimationError(
                  cpu_algorithm_name_, cpu_dev_spec_);
          relative_error_gpu_algorithm =
              hype::Report::instance().getRelativeEstimationError(
                  gpu_algorithm_name_, gpu_dev_spec_);
        }

        std::cout << "Time for Training: " << time_for_training_phase << "ns ("
                  << double(time_for_training_phase) / (1000 * 1000 * 1000)
                  << "s)" << std::endl;

        std::cout << "Time for Workload: "
                  << end_benchmark_timestamp - begin_benchmark_timestamp
                  << "ns ("
                  << double(end_benchmark_timestamp -
                            begin_benchmark_timestamp) /
                         (1000 * 1000 * 1000)
                  << "s)" << std::endl;

        double total_time_cpu = cpu->getTotalProcessingTime();
        double total_time_gpu = gpu->getTotalProcessingTime();
        double total_processing_time_forall_devices =
            total_time_cpu + total_time_gpu;

        unsigned int total_dataset_size_in_bytes = 0;

        for (unsigned int i = 0; i < datasets.size(); i++) {
          total_dataset_size_in_bytes +=
              datasets[i]->getSizeinBytes();  //*sizeof(ElementType);
          // std::cout << "error: missing implementation for setting
          // total_dataset_size_in_bytes" << std::endl;
          // std::exit(0);
        }

        double percentaged_execution_time_on_cpu = 0;
        double percentaged_execution_time_on_gpu = 0;

        if (total_processing_time_forall_devices > 0) {
          percentaged_execution_time_on_cpu =
              total_time_cpu / total_processing_time_forall_devices;
          percentaged_execution_time_on_gpu =
              total_time_gpu / total_processing_time_forall_devices;
        }

        std::cout << "Time for CPU: " << total_time_cpu
                  << "ns \tTime for GPU: " << total_time_gpu << "ns"
                  << std::endl
                  << "CPU Utilization: " << percentaged_execution_time_on_cpu
                  << std::endl
                  << "GPU Utilization: " << percentaged_execution_time_on_gpu
                  << std::endl;

        std::cout << "Relative Error CPU_Algorithm_parallel: "
                  << relative_error_cpu_parallel_algorithm << std::endl;
        std::cout << "Relative Error GPU_Algorithm: "
                  << relative_error_gpu_algorithm << std::endl;

        std::cout << "Total Size of Datasets: " << total_dataset_size_in_bytes
                  << " Byte (" << total_dataset_size_in_bytes / (1024 * 1024)
                  << "MB)" << std::endl;

        std::cout
            << MAX_DATASET_SIZE_IN_MB_ << "\t" << NUMBER_OF_DATASETS_ << "\t"
            << NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ << "\t" << sched_config_
            << "\t" << total_dataset_size_in_bytes << "\t" << RANDOM_SEED_
            << "\t" << stemod_optimization_criterion_ << "\t"
            << stemod_statistical_method_ << "\t"
            << stemod_recomputation_heuristic_ << "\t"
            << hype::core::Runtime_Configuration::instance()
                   .getMaximalReadyQueueLength()
            << "\t"
            << hype::core::Runtime_Configuration::instance().getHistoryLength()
            << "\t"
            << hype::core::Runtime_Configuration::instance()
                   .getRecomputationPeriod()
            << "\t"
            << hype::core::Runtime_Configuration::instance().getTrainingLength()
            << "\t"
            << hype::core::Runtime_Configuration::instance()
                   .getOutlinerThreshold()
            << "\t"
            << hype::core::Runtime_Configuration::instance()
                   .getMaximalSlowdownOfNonOptimalAlgorithm()
            << "\t" << end_benchmark_timestamp - begin_benchmark_timestamp
            << "\t" << time_for_training_phase << "\t" << total_time_cpu << "\t"
            << total_time_gpu << "\t" << percentaged_execution_time_on_cpu
            << "\t" << percentaged_execution_time_on_gpu << "\t"
            << relative_error_cpu_parallel_algorithm << "\t"
            << relative_error_gpu_algorithm << std::endl;

        std::fstream file("benchmark_results.log",
                          std::ios_base::out | std::ios_base::app);

        file.seekg(0,
                   std::ios::end);  // put the "cursor" at the end of the file
        unsigned int file_length =
            file.tellg();  // find the position of the cursor

        if (file_length == 0) {  // if file empty, write header
          file << "MAX_DATASET_SIZE_IN_MB_"
               << "\t"
               << "NUMBER_OF_DATASETS_"
               << "\t"
               << "NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_"
               << "\t"
               << "sched_config_"
               << "\t"
               << "total_size_of_datasets_in_bytes"
               << "\t"
               << "RANDOM_SEED_"
               << "\t"
               << "stemod_optimization_criterion_"
               << "\t"
               << "stemod_statistical_method_"
               << "\t"
               << "stemod_recomputation_heuristic_"
               << "\t"
               << "stemod_maximal_ready_queue_length"
               << "\t"
               << "stemod_history_length"
               << "\t"
               << "stemod_recomputation_period"
               << "\t"
               << "stemod_length_of_training_phase"
               << "\t"
               << "stemod_outliner_threshold_in_percent"
               << "\t"
               << "stemod_maximal_slowdown_of_non_optimal_algorithm"
               << "\t"
               << "workload_execution_time_in_ns"
               << "\t"
               << "execution_time_training_only_in_ns"
               << "\t"
               << "total_time_cpu"
               << "\t"
               << "total_time_gpu"
               << "\t"
               << "spent_time_on_cpu_in_percent"
               << "\t"
               << "spent_time_on_gpu_in_percent"
               << "\t"
               << "average_estimation_error_CPU_Algorithm_parallel"
               << "\t"
               << "average_estimation_error_GPU_Algorithm" << std::endl;
        }

        file
            << MAX_DATASET_SIZE_IN_MB_ << "\t" << NUMBER_OF_DATASETS_ << "\t"
            << NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ << "\t" << sched_config_
            << "\t" << total_dataset_size_in_bytes << "\t" << RANDOM_SEED_
            << "\t" << stemod_optimization_criterion_ << "\t"
            << stemod_statistical_method_ << "\t"
            << stemod_recomputation_heuristic_ << "\t"
            << hype::core::Runtime_Configuration::instance()
                   .getMaximalReadyQueueLength()
            << "\t"
            << hype::core::Runtime_Configuration::instance().getHistoryLength()
            << "\t"
            << hype::core::Runtime_Configuration::instance()
                   .getRecomputationPeriod()
            << "\t"
            << hype::core::Runtime_Configuration::instance().getTrainingLength()
            << "\t"
            << hype::core::Runtime_Configuration::instance()
                   .getOutlinerThreshold()
            << "\t"
            << hype::core::Runtime_Configuration::instance()
                   .getMaximalSlowdownOfNonOptimalAlgorithm()
            << "\t" << end_benchmark_timestamp - begin_benchmark_timestamp
            << "\t" << time_for_training_phase << "\t" << total_time_cpu << "\t"
            << total_time_gpu << "\t" << percentaged_execution_time_on_cpu
            << "\t" << percentaged_execution_time_on_gpu << "\t"
            << relative_error_cpu_parallel_algorithm << "\t"
            << relative_error_gpu_algorithm << std::endl;

        file.close();
        return 0;
      }
    };

  }  // end namespace queryprocessing
}  // end namespace hype
