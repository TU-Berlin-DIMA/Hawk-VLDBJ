/*
 * File:   generic_operator_benchmark.hpp
 * Author: sebastian
 *
 * Created on 6. Oktober 2013, 19:42
 */

#ifndef GENERIC_OPERATOR_BENCHMARK_HPP
#define GENERIC_OPERATOR_BENCHMARK_HPP

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <stdint.h>
#include <sys/mman.h>

#include <config/configuration.hpp>
#include <hype.hpp>
#include <util/architecture.hpp>
//#include <util/hardware_detector.hpp>
#include <query_processing/logical_query_plan.hpp>
#include <query_processing/node.hpp>
#include <query_processing/operator.hpp>
#include <query_processing/processing_device.hpp>

#include <tbb/parallel_sort.h>
#include <tbb/task_scheduler_init.h>

#include <boost/chrono.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/program_options.hpp>
#include <boost/random.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/thread.hpp>

#include "core/scheduler.hpp"

namespace hype {

  namespace queryprocessing {

    boost::program_options::options_description&
    getGlobalBenchmarkOptionDescription() {
      static boost::program_options::options_description
          global_benchmark_option_description("allowed options");
      static bool initialized = false;
      if (!initialized) {
        // add command line parameters
        global_benchmark_option_description.add_options()(
            "help", "produce help message")(
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
        initialized = true;
      }
      return global_benchmark_option_description;
    }

    // enum SchedulingConfiguration{CPU_ONLY,GPU_ONLY,HYBRID};
    template <typename Type>
    class Generic_Operation_Benchmark {
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

      typedef std::vector<DeviceSpecification> DeviceSpecifications;
      typedef std::vector<AlgorithmSpecification> AlgorithmSpecifications;
      typedef std::vector<hype::queryprocessing::ProcessingDevicePtr>
          ProcessingDevices;

      Generic_Operation_Benchmark(
          const std::string& operation_name,
          // alg specs all have to have the same operation!
          const AlgorithmSpecifications& alg_specs_,
          const DeviceSpecifications& dev_specs_)
          : operation_name_(operation_name),
            MAX_DATASET_SIZE_IN_MB_(
                1),  //(10), //MB  //(10*1000*1000)/sizeof(int), //1000000,
            NUMBER_OF_DATASETS_(10),                     // 3, //100,
            NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_(30),  // 3, //1000,
            RANDOM_SEED_(0),
            sched_config_(CPU_ONLY),  //(HYBRID), //CPU_ONLY,GPU_ONLY,HYBRID
            stemod_optimization_criterion_("Response Time"),
            stemod_statistical_method_("Least Squares 1D"),
            stemod_recomputation_heuristic_("Periodic Recomputation"),
            // cpu( DeviceSpecification (hype::PD0,hype::CPU,hype::PD_Memory_0)
            // ),
            // gpu( DeviceSpecification (hype::PD1,hype::GPU,hype::PD_Memory_1)
            // ),
            proc_devs_(),
            dev_specs_(dev_specs_),
            alg_specs_(alg_specs_),
            datasets(),
            operator_queries_(),
            rng_(),
            desc_(getGlobalBenchmarkOptionDescription())  //,
      // operator_map_()
      {
        //            //ensure we have enough slots to train all algorithms on
        //            each processing device
        //            assert(hype::core::Runtime_Configuration::instance().getMaximalReadyQueueLength()>=alg_specs_.size()*hype::core::Runtime_Configuration::instance().getTrainingLength());

        for (unsigned int i = 0; i < dev_specs_.size(); ++i) {
          for (unsigned int j = 0; j < alg_specs_.size(); ++j) {
            Scheduler::instance().addAlgorithm(alg_specs_[j], dev_specs_[i]);
          }
        }

        // init array of active processing devices
        for (unsigned int i = 0; i < dev_specs_.size(); ++i) {
          ProcessingDevicePtr proc_dev_ptr =
              core::Scheduler::instance()
                  .getProcessingDevices()
                  .getProcessingDevice(dev_specs_[i].getProcessingDeviceID());
          assert(proc_dev_ptr != NULL);
          // if the following assertion is true, we can use the device id for
          // fast lookup!
          assert(i == dev_specs_[i].getProcessingDeviceID());
          proc_devs_.push_back(proc_dev_ptr);
        }

        if (!alg_specs_.empty()) {
          for (unsigned int i = 0; i < alg_specs_.size(); ++i) {
            // check whether the algorithm specification belongs to the
            // specified operation
            assert(operation_name_ == alg_specs_[i].getOperationName());
          }
        }
        //            //add command line parameters
        //            desc_.add_options()
        //            ("help", "produce help message")
        //            ("number_of_datasets",
        //            boost::program_options::value<unsigned int>(), "set the
        //            number of data sets for workload")
        //            ("number_of_operations",
        //            boost::program_options::value<unsigned int>(), "set the
        //            number of operations in workload")
        //            ("max_dataset_size_in_MB",
        //            boost::program_options::value<unsigned int>(), "set the
        //            maximal dataset size in MB")
        //                 //("ready_queue_length",
        //                 boost::program_options::value<unsigned int>(), "set
        //                 the queue length of operators that may be
        //                 concurrently scheduled (clients are blocked on a
        //                 processing device)")
        //            ("scheduling_method",
        //            boost::program_options::value<std::string>(), "set the
        //            decision model (CPU_ONLY, GPU_ONLY, HYBRID)")
        //            ("random_seed", boost::program_options::value<unsigned
        //            int>(), "seed to use before for generating datasets and
        //            operation workload")
        //            ("optimization_criterion",
        //            boost::program_options::value<std::string>(), "set the
        //            decision models optimization_criterion for all
        //            algorithms")
        //            ("statistical_method",
        //            boost::program_options::value<std::string>(), "set the
        //            decision models statistical_method for all algorithms")
        //            ("recomputation_heuristic",
        //            boost::program_options::value<std::string>(), "set the
        //            decision models recomputation_heuristic for all
        //            algorithms")
        //            ;
      }

      // RandomNumberGenerator random_number_generator_;
      std::string operation_name_;
      std::string algorithm_name_;

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

      ProcessingDevices proc_devs_;
      DeviceSpecifications dev_specs_;
      AlgorithmSpecifications alg_specs_;

      std::vector<Type> datasets;
      std::vector<TypedNodePtr> operator_queries_;

      boost::mt19937 rng_;  // produces randomness out of thin air
      boost::program_options::options_description
          desc_;  // see pseudo-random number generators

      boost::mt19937& getRandomNumberGenerator() { return rng_; }

      boost::program_options::options_description& getOptionsDescription() {
        return desc_;
      }

      // Physical_Operator_Map_Ptr operator_map_;

      static uint64_t getTimestamp() {
        using namespace boost::chrono;

        high_resolution_clock::time_point tp = high_resolution_clock::now();
        nanoseconds dur = tp.time_since_epoch();

        return (uint64_t)dur.count();
      }

      virtual TypedNodePtr generate_logical_operator(Type dataset) = 0;

      // virtual vector<TypedNodePtr> createOperatorQueries() = 0;

      virtual Type generate_dataset(unsigned int size_in_number_of_bytes) = 0;

      // virtual destructor
      virtual ~Generic_Operation_Benchmark() {}

      std::vector<Type> generate_random_datasets(
          unsigned int max_size_of_dataset_in_byte,
          unsigned int number_of_datasets) {
        std::vector<Type> datasets;
        // first, generate dataset of full possible size, then decrease it with
        // each loop according to a value tic, until the last dataset size is
        // only tic
        unsigned int tic = max_size_of_dataset_in_byte / number_of_datasets;
        for (unsigned int i = 0; i < number_of_datasets; i++) {
          Type vec_ptr = this->generate_dataset(max_size_of_dataset_in_byte -
                                                i * tic);  //(unsigned int)
          //(rand()%max_size_in_number_of_elements) );
          // assert(vec_ptr!=NULL);
          datasets.push_back(vec_ptr);
        }
        return datasets;
      }

      int setup(int argc, char* argv[]) {
        // we don't want the OS to swap out our data to disc that's why we lock
        // it
        mlockall(MCL_CURRENT | MCL_FUTURE);

        //	tbb::task_scheduler_init init(8);

        // Declare the supported options.
        // boost::program_options::options_description desc("Allowed options");
        // desc.add_options()
        //    ("help", "produce help message")
        //    ("number_of_datasets", boost::program_options::value<unsigned
        //    int>(), "set the number of data sets for workload")
        //    ("number_of_operations", boost::program_options::value<unsigned
        //    int>(), "set the number of operations in workload")
        //    ("max_dataset_size_in_MB", boost::program_options::value<unsigned
        //    int>(), "set the maximal dataset size in MB")
        //	 //("ready_queue_length", boost::program_options::value<unsigned
        // int>(), "set the queue length of operators that may be concurrently
        // scheduled (clients are blocked on a processing device)")
        //    ("scheduling_method",
        //    boost::program_options::value<std::string>(), "set the decision
        //    model (CPU_ONLY, GPU_ONLY, HYBRID)")
        //    ("random_seed", boost::program_options::value<unsigned int>(),
        //    "seed to use before for generating datasets and operation
        //    workload")
        //    ("optimization_criterion",
        //    boost::program_options::value<std::string>(), "set the decision
        //    models optimization_criterion for all algorithms")
        //    ("statistical_method",
        //    boost::program_options::value<std::string>(), "set the decision
        //    models statistical_method for all algorithms")
        //    ("recomputation_heuristic",
        //    boost::program_options::value<std::string>(), "set the decision
        //    models recomputation_heuristic for all algorithms")
        //;

        boost::program_options::variables_map vm;
        boost::program_options::store(
            boost::program_options::parse_command_line(argc, argv, desc_), vm);
        boost::program_options::notify(vm);

        if (vm.count("help")) {
          std::cout << desc_ << "\n";
          exit(0);
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

        std::cout << "Generating Data sets..." << std::endl;
        std::cout << "Estimated RAM usage: "
                  << estimated_ram_usage_in_byte / (1024 * 1024) << "MB"
                  << std::endl;
        if ((estimated_ram_usage_in_byte / (1024 * 1024)) > 1024 * 3.7 &&
            util::getArchitecture() == Architecture_32Bit) {
          std::cout << "Warning: Memory for Datasets to generate exceeds 32 "
                       "bit adress space! ("
                    << estimated_ram_usage_in_byte / (1024 * 1024) << "MB)"
                    << std::endl;  // << "Exiting..." << std::endl;
          // std::exit(-1);
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
          Type dataset = datasets[index];
          TypedNodePtr op = generate_logical_operator(dataset);
          assert(op != NULL);
          operator_queries_.push_back(op);
        }

        // std::copy(query_indeces.begin(), query_indeces.end(),
        // std::ostream_iterator<unsigned int>(std::cout, "\n"));

        // setup HyPE
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

        // typedef std::vector<DeviceSpecification> DeviceSpecifications
        // dev_specs =
        // CoGaDB::HardwareDetector::instance().getDeviceSpecifications();

        for (unsigned int i = 0; i < dev_specs_.size(); ++i) {
          for (unsigned int j = 0; j < alg_specs_.size(); ++j) {
            // Scheduler::instance().addAlgorithm(alg_specs_[j],dev_specs_[i]);
            /* set statistical method */
            if (!hype::Scheduler::instance().setStatisticalMethod(
                    alg_specs_[j].getAlgorithmName(), dev_specs_[i],
                    stemod_statistical_method_)) {
              std::cout << "Error setting StatisticalMethod "
                        << stemod_statistical_method_ << " for algorithm: "
                        << alg_specs_[j].getAlgorithmName() << std::endl;
              return -1;
            }
            /* set recomputation heuristic */
            if (!hype::Scheduler::instance().setRecomputationHeuristic(
                    alg_specs_[j].getAlgorithmName(), dev_specs_[i],
                    stemod_recomputation_heuristic_)) {
              std::cout << "Error setting RecomputationHeuristic "
                        << stemod_statistical_method_ << " for algorithm: "
                        << alg_specs_[j].getAlgorithmName() << std::endl;
              return -1;
            }
          }
        }

        // ensure we have enough slots to train all algorithms on each
        // processing device
        assert(hype::core::Runtime_Configuration::instance()
                   .getMaximalReadyQueueLength() >=
               alg_specs_.size() *
                   hype::core::Runtime_Configuration::instance()
                       .getTrainingLength());
        // we need at least so many operators in the workload, so we can
        // compelte our training phase
        assert(this->NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ >=
               alg_specs_.size() *
                   hype::core::Runtime_Configuration::instance()
                       .getTrainingLength());

        /*start processing devices*/
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          assert(proc_devs_[i] != NULL);
          proc_devs_[i]->start();
        }

        return 0;
      }

      int run() {
        std::cout << "Starting Benchmark..." << std::endl;

        uint64_t begin_benchmark_timestamp = getTimestamp();
        uint64_t end_training_timestamp = 0;

        for (unsigned int i = 0; i < NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_;
             i++) {
          TypedNodePtr current_operator = operator_queries_[i];
          TypedNodePtr scan =
              boost::static_pointer_cast<typename TypedNodePtr::element_type>(
                  current_operator->getLeft());

          // std::cout << "RUN: " << i << "/" <<
          // NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ << std::endl;

          if (sched_config_ == HYBRID) {  // CPU_ONLY,GPU_ONLY,HYBRID)
            // cout << "scheduling operator " << i << std::endl;
            const unsigned int number_of_algorithms_per_device =
                alg_specs_.size();
            const unsigned int number_of_devices = dev_specs_.size();
            const unsigned int total_number_of_algorithms =
                number_of_algorithms_per_device * number_of_devices;
            const unsigned int number_of_training_operations =
                (hype::core::Runtime_Configuration::instance()
                     .getTrainingLength() *
                 total_number_of_algorithms) +
                1;  //*number of algortihms per operation (2)
            if (number_of_training_operations == i) {
              if (!hype::core::quiet)
                std::cout << "waiting for training to complete" << std::endl;
              // wait until training operations finished
              // make variable volatile so the compiler does not optimize it
              // away
              volatile bool continue_loop = true;
              while (continue_loop) {
                continue_loop = false;
                // reduce overhead of busy waiting
                boost::this_thread::sleep(boost::posix_time::microseconds(20));
                for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
                  assert(proc_devs_[i] != NULL);
                  // if at least one processing device is not finished, wait and
                  // continue loop!
                  if (!proc_devs_[i]->isIdle()) continue_loop = true;
                }
              }
              //					while(!cpu->isIdle() ||
              //! gpu->isIdle()){
              //						//std::cout <<
              //"stat: cpu " << !cpu->isIdle() << " gpu " << !gpu->isIdle() <<
              // std::endl;
              //						boost::this_thread::sleep(boost::posix_time::microseconds(20));
              //					}
              end_training_timestamp = getTimestamp();
              // cout << "stat: cpu " << !cpu->isIdle() << " gpu " <<
              // !gpu->isIdle() << std::endl;
              // if(!hype::core::quiet)
              std::cout << "training completed! Time: "
                        << end_training_timestamp - begin_benchmark_timestamp
                        << "ns ("
                        << double(end_training_timestamp -
                                  begin_benchmark_timestamp) /
                               (1000 * 1000 * 1000)
                        << "s)" << std::endl;
            }

            TypedOperatorPtr phy_scan;  // TypedOperatorPtr phy_scan =
            // scan->getOptimalOperator(TypedOperatorPtr(),TypedOperatorPtr());
            if (scan)
              phy_scan = scan->getOptimalOperator(TypedOperatorPtr(),
                                                  TypedOperatorPtr());
            TypedOperatorPtr phy_op = current_operator->getOptimalOperator(
                phy_scan, TypedOperatorPtr());

            ProcessingDeviceID dev_id =
                phy_op->getDeviceSpecification().getProcessingDeviceID();
            // ensure no array out of bounds error can occur
            assert(dev_id < proc_devs_.size());
            // ensure array contains valid pointer to ProcessingDevice
            assert(proc_devs_[dev_id] != NULL);
            proc_devs_[dev_id]->addOperator(phy_op);

          } else if (sched_config_ == CPU_ONLY) {
            TypedOperatorPtr phy_scan;
            if (scan)
              phy_scan = scan->getOptimalOperator(TypedOperatorPtr(),
                                                  TypedOperatorPtr());
            TypedOperatorPtr phy_op = current_operator->getOptimalOperator(
                phy_scan, TypedOperatorPtr(), hype::CPU_ONLY);
            assert(phy_op->getDeviceSpecification().getDeviceType() == CPU);
            // cpu->addOperator(phy_op);
            // execute algorithm on the CPU specified by the scheduling decision
            ProcessingDeviceID dev_id =
                phy_op->getDeviceSpecification().getProcessingDeviceID();
            // ensure no array out of bounds error can occur
            assert(dev_id < proc_devs_.size());
            // ensure array contains valid pointer to ProcessingDevice
            assert(proc_devs_[dev_id] != NULL);
            proc_devs_[dev_id]->addOperator(phy_op);

          } else if (sched_config_ == GPU_ONLY) {
            TypedOperatorPtr phy_scan;
            if (scan)
              phy_scan = scan->getOptimalOperator(TypedOperatorPtr(),
                                                  TypedOperatorPtr());
            TypedOperatorPtr phy_op = current_operator->getOptimalOperator(
                phy_scan, TypedOperatorPtr(), hype::CPU_ONLY);
            assert(phy_op->getDeviceSpecification().getDeviceType() == GPU);
            // cpu->addOperator(phy_op);
            // execute algorithm on the CPU specified by the scheduling decision
            ProcessingDeviceID dev_id =
                phy_op->getDeviceSpecification().getProcessingDeviceID();
            // ensure no array out of bounds error can occur
            assert(dev_id < proc_devs_.size());
            // ensure array contains valid pointer to ProcessingDevice
            assert(proc_devs_[dev_id] != NULL);
            proc_devs_[dev_id]->addOperator(phy_op);
          }
        }

        //	boost::this_thread::sleep( boost::posix_time::seconds(3) );

        //	cpu->stop();
        //	gpu->stop();

        //	while(!cpu->isIdle() || !gpu->isIdle()){
        //		boost::this_thread::sleep(boost::posix_time::microseconds(20));
        //	}
        // wait until training operations finished
        // make variable volatile so the compiler does not optimize it away
        volatile bool continue_loop = true;
        while (continue_loop) {
          continue_loop = false;
          // reduce overhead of busy waiting
          boost::this_thread::sleep(boost::posix_time::microseconds(20));
          for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
            assert(proc_devs_[i] != NULL);
            // if at least one processing device is not finished, wait and
            // continue loop!
            if (!proc_devs_[i]->isIdle()) continue_loop = true;
          }
        }

        uint64_t end_benchmark_timestamp = getTimestamp();
        // std::cout << "stat: cpu " << !cpu->isIdle() << " gpu " <<
        // !gpu->isIdle() << std::endl;
        // if(!core::quiet)
        {
          std::cout << "System Status:" << std::endl;
          for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
            assert(proc_devs_[i] != NULL);
            ProcessingDeviceID proc_dev_id =
                proc_devs_[i]->getProcessingDeviceID();
            queryprocessing::VirtualProcessingDevicePtr virt_dev_ptr =
                core::Scheduler::instance()
                    .getProcessingDevices()
                    .getVirtualProcessingDevice(proc_dev_id);
            queryprocessing::ProcessingDevicePtr phy_dev_ptr =
                core::Scheduler::instance()
                    .getProcessingDevices()
                    .getProcessingDevice(proc_dev_id);
            assert(virt_dev_ptr != NULL);
            assert(phy_dev_ptr != NULL);
            virt_dev_ptr->print();
            std::cout << "\t"
                      << "Total Processing Time: "
                      << phy_dev_ptr->getTotalProcessingTime() << "ns ("
                      << phy_dev_ptr->getTotalProcessingTime() /
                             (1000 * 1000 * 1000)
                      << "s)" << std::endl;
          }
        }

        std::cout << "[Main Thread] Processing Devices finished..."
                  << std::endl;

        //	cpu->stop();
        //	gpu->stop();
        /*stop processing devices*/
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          assert(proc_devs_[i] != NULL);
          proc_devs_[i]->stop();
        }

        // if one of the following assertiosn are not fulfilled, then abort,
        // because results are rubbish
        assert(end_benchmark_timestamp >= begin_benchmark_timestamp);
        double time_for_training_phase = 0;
        double relative_error_cpu_parallel_algorithm = 0;
        double relative_error_gpu_algorithm = 0;

        // we have #algorithms*#processing devices relative errors to retrieve
        std::vector<double> relative_errors(alg_specs_.size() *
                                            dev_specs_.size());

        if (sched_config_ == HYBRID) {  // a training phase only exists when the
                                        // decision model is used
          assert(end_training_timestamp >= begin_benchmark_timestamp);
          assert(end_benchmark_timestamp >= end_training_timestamp);
          time_for_training_phase =
              end_training_timestamp - begin_benchmark_timestamp;
          for (unsigned int i = 0; i < dev_specs_.size(); ++i) {
            for (unsigned int j = 0; j < alg_specs_.size(); ++j) {
              relative_errors[i * alg_specs_.size() + j] =
                  hype::Report::instance().getRelativeEstimationError(
                      alg_specs_[j].getAlgorithmName(), dev_specs_[i]);
            }
          }
          // relative_error_cpu_parallel_algorithm =
          // hype::Report::instance().getRelativeEstimationError(cpu_algorithm_name_,cpu_dev_spec_);
          // relative_error_gpu_algorithm =
          // hype::Report::instance().getRelativeEstimationError(gpu_algorithm_name_,gpu_dev_spec_);
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

        std::vector<double> proc_dev_execution_times(proc_devs_.size());

        double total_processing_time_forall_devices = 0;
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          proc_dev_execution_times[i] = proc_devs_[i]->getTotalProcessingTime();
          total_processing_time_forall_devices += proc_dev_execution_times[i];
        }

        std::vector<double> percentaged_execution_times(proc_devs_.size());
        // sqrt(sum i=1 .. n: (percentaged_execution_times[i]-(1/n))^2)
        double equal_processing_utilization_error = 0;
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          percentaged_execution_times[i] = proc_dev_execution_times[i] /
                                           total_processing_time_forall_devices;
          equal_processing_utilization_error += std::pow(
              percentaged_execution_times[i] - double(1 / proc_devs_.size()),
              2);
        }
        // equal_processing_utilization_error=std::sqrt(equal_processing_utilization_error);

        //	double total_time_cpu=cpu->getTotalProcessingTime();
        //	double total_time_gpu=gpu->getTotalProcessingTime();
        //	double total_processing_time_forall_devices=total_time_cpu +
        // total_time_gpu;

        unsigned int total_dataset_size_in_bytes = 0;

        for (unsigned int i = 0; i < datasets.size(); i++) {
#ifndef HYPE_GENERIC_BENCHMARK_DATATYPE_IS_POINTER
          total_dataset_size_in_bytes +=
              datasets[i].getSizeinBytes();  //*sizeof(ElementType);
#else
          total_dataset_size_in_bytes +=
              datasets[i]->getSizeinBytes();  //*sizeof(ElementType);
#endif
          // std::cout << "error: missing implementation for setting
          // total_dataset_size_in_bytes" << std::endl;
          // std::exit(0);
        }

        //	double percentaged_execution_time_on_cpu=0;
        //	double percentaged_execution_time_on_gpu=0;
        //
        //	if(total_processing_time_forall_devices>0){
        //		percentaged_execution_time_on_cpu =
        // total_time_cpu/total_processing_time_forall_devices;
        //		percentaged_execution_time_on_gpu =
        // total_time_gpu/total_processing_time_forall_devices;
        //	}

        std::cout << "========================================================="
                     "============="
                  << std::endl;
        std::cout << "HyPE Report:" << std::endl;

        // print total execution times (ns)
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          std::cout << "Time for "
                    << hype::util::getName(dev_specs_[i].getDeviceType()) << "_"
                    << (int)dev_specs_[i].getProcessingDeviceID() << ": "
                    << proc_dev_execution_times[i] << "ns" << std::endl;
        }
        std::cout << "---------------------------------------------------------"
                     "-------------"
                  << std::endl;
        // print percentaged execution time w.r.t. execution time spend on all
        // processing devices
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          std::cout << "Utilization of "
                    << hype::util::getName(dev_specs_[i].getDeviceType()) << "_"
                    << (int)dev_specs_[i].getProcessingDeviceID() << ": "
                    << percentaged_execution_times[i] * 100 << "%" << std::endl;
        }
        std::cout << "Equal Processing Device Utilization Error: "
                  << equal_processing_utilization_error
                  << " (ideal: 0, worst: ?)" << std::endl;
        std::cout << "---------------------------------------------------------"
                     "-------------"
                  << std::endl;
        // print relative estimation errors
        for (unsigned int i = 0; i < dev_specs_.size(); ++i) {
          for (unsigned int j = 0; j < alg_specs_.size(); ++j) {
            std::cout << "Relative Estimation Error "
                      << alg_specs_[j].getAlgorithmName() << " on "
                      << hype::util::getName(dev_specs_[i].getDeviceType())
                      << "_" << (int)dev_specs_[i].getProcessingDeviceID()
                      << ": "
                      << relative_errors[i * alg_specs_.size() + j] * 100 << "%"
                      << std::endl;
          }
        }
        std::cout << "---------------------------------------------------------"
                     "-------------"
                  << std::endl;
        //	std::cout << "Time for CPU: " <<  total_time_cpu  << "ns \tTime
        // for GPU: " << total_time_gpu << "ns" << std::endl
        //		  << "CPU Utilization: " <<
        // percentaged_execution_time_on_cpu << std::endl
        //		  << "GPU Utilization: " <<
        // percentaged_execution_time_on_gpu << std::endl;
        //
        //	std::cout << "Relative Error CPU_Algorithm_parallel: " <<
        // relative_error_cpu_parallel_algorithm << std::endl;
        //	std::cout << "Relative Error GPU_Algorithm: " <<
        // relative_error_gpu_algorithm	 << std::endl;

        std::cout << "Total Size of Datasets: " << total_dataset_size_in_bytes
                  << " Byte (" << total_dataset_size_in_bytes / (1024 * 1024)
                  << "MB)" << std::endl;
        std::cout << "---------------------------------------------------------"
                     "-------------"
                  << std::endl;

        std::stringstream str_stream;
        str_stream
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
            << "\t" << time_for_training_phase << "\t" << proc_devs_.size()
            << "\t" << alg_specs_.size() << "\t";
        // print total execution times (ns)
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          str_stream << proc_dev_execution_times[i] << "\t";
        }
        // print percentaged execution time w.r.t. execution time spend on all
        // processing devices
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          str_stream << percentaged_execution_times[i] << "\t";
        }
        // print relative estimation errors
        for (unsigned int i = 0; i < dev_specs_.size(); ++i) {
          for (unsigned int j = 0; j < alg_specs_.size(); ++j) {
            str_stream << relative_errors[i * alg_specs_.size() + j] << "\t";
          }
        }
        //		<< total_time_cpu << "\t"
        //		<< total_time_gpu << "\t"
        //		<< percentaged_execution_time_on_cpu << "\t"
        //		<< percentaged_execution_time_on_gpu << "\t"
        //		<< relative_error_cpu_parallel_algorithm << "\t"
        //		<< relative_error_gpu_algorithm
        // str_stream << std::endl;

        std::string result_line = str_stream.str();

        std::stringstream str_stream_header_line;

        str_stream_header_line
            << "MAX_DATASET_SIZE_IN_MB_"
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
            << "#processing_devices"
            << "\t"
            << "#algortihms_per_device"
            << "\t";
        // print total execution times (ns)
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          str_stream_header_line
              << "total_time_"
              << hype::util::getName(dev_specs_[i].getDeviceType()) << "_"
              << (int)dev_specs_[i].getProcessingDeviceID() << "\t";
        }
        // print percentaged execution time w.r.t. execution time spend on all
        // processing devices
        for (unsigned int i = 0; i < proc_devs_.size(); ++i) {
          str_stream_header_line
              << "spent_time_on_"
              << hype::util::getName(dev_specs_[i].getDeviceType()) << "_"
              << (int)dev_specs_[i].getProcessingDeviceID() << "_in_percent"
              << "\t";
        }
        // print relative estimation errors
        for (unsigned int i = 0; i < dev_specs_.size(); ++i) {
          for (unsigned int j = 0; j < alg_specs_.size(); ++j) {
            str_stream_header_line
                << "average_estimation_error_"
                << alg_specs_[j].getAlgorithmName() << "_on_"
                << hype::util::getName(dev_specs_[i].getDeviceType()) << "_"
                << (int)dev_specs_[i].getProcessingDeviceID() << "\t";
          }
        }
        //		<< "total_time_cpu"  << "\t"
        //		<< "total_time_gpu"  << "\t"
        //		<< "spent_time_on_cpu_in_percent"  << "\t"
        //		<< "spent_time_on_gpu_in_percent"  << "\t"
        //		<< "average_estimation_error_CPU_Algorithm_parallel" <<
        //"\t"
        //		<< "average_estimation_error_GPU_Algorithm"
        // str_stream_header_line << std::endl;

        std::string result_header_line = str_stream_header_line.str();

        std::cout << result_header_line << std::endl;
        std::cout << result_line << std::endl;
        std::cout << "========================================================="
                     "============="
                  << std::endl;

        std::fstream file("benchmark_results.log",
                          std::ios_base::out | std::ios_base::app);

        file.seekg(0,
                   std::ios::end);  // put the "cursor" at the end of the file
        unsigned int file_length =
            file.tellg();  // find the position of the cursor

        if (file_length == 0) {  // if file empty, write header
          file << result_header_line << std::endl;
          // file << "MAX_DATASET_SIZE_IN_MB_" << "\t"
          //		<< "NUMBER_OF_DATASETS_" << "\t"
          //		<< "NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_" << "\t"
          //		<< "sched_config_" << "\t"
          //		<< "total_size_of_datasets_in_bytes" << "\t"
          //		<< "RANDOM_SEED_" << "\t"
          //		<< "stemod_optimization_criterion_" << "\t"
          //		<< "stemod_statistical_method_" << "\t"
          //		<< "stemod_recomputation_heuristic_" << "\t"
          //		<< "stemod_maximal_ready_queue_length" << "\t"
          //		<< "stemod_history_length" << "\t"
          //		<< "stemod_recomputation_period" << "\t"
          //		<< "stemod_length_of_training_phase" << "\t"
          //		<< "stemod_outliner_threshold_in_percent" << "\t"
          //		<< "stemod_maximal_slowdown_of_non_optimal_algorithm" <<
          //"\t"
          //		<< "workload_execution_time_in_ns" << "\t"
          //		<< "execution_time_training_only_in_ns" << "\t"	;
          //                //print total execution times (ns)
          //                for(unsigned int i=0;i<proc_devs_.size();++i){
          //                        cout << "total_time_" <<
          //                        hype::util::getName(dev_specs_[i].getDeviceType())
          //                        <<"_"<<
          //                        (int)dev_specs_[i].getProcessingDeviceID()
          //                        << "\t";
          //                }
          //                //print percentaged execution time w.r.t. execution
          //                time spend on all processing devices
          //                for(unsigned int i=0;i<proc_devs_.size();++i){
          //                       cout << "spent_time_on_"  <<
          //                       hype::util::getName(dev_specs_[i].getDeviceType())
          //                       <<"_"<<
          //                       (int)dev_specs_[i].getProcessingDeviceID() <<
          //                       "_in_percent" << "\t";
          //                }
          //                //print relative estimation errors
          //                for(unsigned int i=0;i<dev_specs_.size();++i){
          //                     for(unsigned int j=0;j<alg_specs_.size();++j){
          //                          cout << "average_estimation_error_" <<
          //                          alg_specs_[j].getAlgorithmName() << "_on_"
          //                          <<
          //                          hype::util::getName(dev_specs_[i].getDeviceType())
          //                          <<"_" <<
          //                          (int)dev_specs_[i].getProcessingDeviceID()
          //                          << "\t";
          //                     }
          //                }
          ////		<< "total_time_cpu"  << "\t"
          ////		<< "total_time_gpu"  << "\t"
          ////		<< "spent_time_on_cpu_in_percent"  << "\t"
          ////		<< "spent_time_on_gpu_in_percent"  << "\t"
          ////		<< "average_estimation_error_CPU_Algorithm_parallel" <<
          ///"\t"
          ////		<< "average_estimation_error_GPU_Algorithm"
          //		cout << std::endl;
        }

        file << result_line << std::endl;

        // file << MAX_DATASET_SIZE_IN_MB_ << "\t"
        //		<< NUMBER_OF_DATASETS_ << "\t"
        //		<< NUMBER_OF_SORT_OPERATIONS_IN_WORKLOAD_ << "\t"
        //		<< sched_config_ << "\t"
        //		<< total_dataset_size_in_bytes << "\t"
        //		<< RANDOM_SEED_ << "\t"
        //		<< stemod_optimization_criterion_ << "\t"
        //		<< stemod_statistical_method_ << "\t"
        //		<< stemod_recomputation_heuristic_ << "\t"
        //		<<
        // hype::core::Runtime_Configuration::instance().getMaximalReadyQueueLength()
        //<< "\t"
        //		<<
        // hype::core::Runtime_Configuration::instance().getHistoryLength() <<
        //"\t"
        //		<<
        // hype::core::Runtime_Configuration::instance().getRecomputationPeriod()
        //<< "\t"
        //		<<
        // hype::core::Runtime_Configuration::instance().getTrainingLength() <<
        //"\t"
        //		<<
        // hype::core::Runtime_Configuration::instance().getOutlinerThreshold()<<
        //"\t"
        //		<<
        // hype::core::Runtime_Configuration::instance().getMaximalSlowdownOfNonOptimalAlgorithm()
        //<< "\t"
        //		<< end_benchmark_timestamp-begin_benchmark_timestamp <<
        //"\t"
        //		<< time_for_training_phase << "\t"
        //		<< total_time_cpu << "\t"
        //		<< total_time_gpu << "\t"
        //		<< percentaged_execution_time_on_cpu << "\t"
        //		<< percentaged_execution_time_on_gpu  << "\t"
        //		<< relative_error_cpu_parallel_algorithm << "\t"
        //		<< relative_error_gpu_algorithm
        //		<< std::endl;

        file.close();
        return 0;
      }
    };

  }  // end namespace queryprocessing
}  // end namespace hype

#endif /* GENERIC_OPERATOR_BENCHMARK_HPP */
