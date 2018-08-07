
#include <core/variable_manager.hpp>
#include <optimizer/optimizer.hpp>
#include <parser/client.hpp>
#include <query_compilation/code_generator.hpp>
#include <query_compilation/query_context.hpp>
#include <query_processing/query_processor.hpp>
#include <util/genetic_exploration.hpp>
#include <util/query_processing.hpp>
#include <util/variant_configurator.hpp>
#include <util/variant_measurement.hpp>

#include "query_compilation/code_generators/code_generator_utils.hpp"

#include <vector>

#ifdef COGADB_MEASURE_ENERGY_CONSUMPTION
#include <IntelPerformanceCounterMonitorV2.7/cpucounters.h>
#endif

#ifdef PERSEUS_FOUND
#include <query_compilation/variant_tuning/multi_variant_pipeline.hpp>
#endif

namespace CoGaDB {

#ifdef PERSEUS_FOUND
const TablePtr executeQueryPlanWithPerseus(
    query_processing::LogicalQueryPlanPtr logical_plan, ClientPtr client,
    CodeGeneratorType generator_type, VariantIterator& variant_iterator) {
  auto pipeline =
      boost::make_shared<MultiVariantPipeline>(*logical_plan, variant_iterator);
  auto context = createQueryContext();
  auto result = CoGaDB::execute(pipeline, context);
  return result;
}
#endif

#define OUTLIER_DIFF_THRESHOLD 2

const std::pair<TablePtr, VariantMeasurement> executeQueryPlanWithCompiler(
    query_processing::LogicalQueryPlanPtr log_plan, ClientPtr client,
    CodeGeneratorType code_gen_type, const Variant* variant) {
  Timestamp begin_total_time = getTimestamp();
  TablePtr result;
  VariantConfigurator vc;
  /* setup variant, if the caller gave us one */
  if (variant) {
    vc(*variant);
  }
  ProjectionParam project_param;
  CodeGeneratorPtr code_gen = createCodeGenerator(code_gen_type, project_param);
  QueryContextPtr context = createQueryContext();
  context->markAsOriginalContext();
  log_plan->getRoot()->produce(code_gen, context);
  PipelinePtr pipeline = code_gen->compile();
  assert(pipeline != NULL);
  result = CoGaDB::execute(pipeline, context);
  Timestamp end_total_time = getTimestamp();

  double total_compile_time_s = context->getCompileTimeSec();
  double total_host_compile_time_s = context->getHostCompileTimeSec();
  double total_kernel_compile_time_s = context->getKernelCompileTimeSec();
  double total_execution_time_s =
      double(end_total_time - begin_total_time) / (1000 * 1000 * 1000) -
      context->getCompileTimeSec();
  double pipeline_execution_time_s = context->getExecutionTimeSec();
  double overhead_time_s =
      ((double(end_total_time - begin_total_time) / (1000 * 1000 * 1000)) -
       context->getCompileTimeSec()) -
      context->getExecutionTimeSec();

  VariantMeasurement vm(
      true, total_execution_time_s, pipeline_execution_time_s,
      total_compile_time_s,
      total_kernel_compile_time_s, /* kernel compilation time */
      total_host_compile_time_s,   /* host compilation time */
      overhead_time_s);

  return std::make_pair(result, vm);
}

Variant fastest_variant;

Variant getCurrentVariant(VariantIterator& itr) {
  Variant variant;

  for (auto& dimension : itr.getSortedDimensions()) {
    variant[dimension.getName()] =
        VariableManager::instance().getVariableValueString(dimension.getName());
  }

  return variant;
}

Variant getStartVariantExloration(VariantIterator& itr) {
  Variant variant;
  VariantIterator::FlattenedDimensions flattened_dimensions =
      itr.getFlattenedDimensions();
  for (auto& dimension : flattened_dimensions) {
    variant[dimension.first] =
        VariableManager::instance().getVariableValueString(dimension.first);
  }
  return variant;
}

const TablePtr exploreVariants(const std::string& variant_exploration_mode,
                               CodeGeneratorType code_gen_type,
                               query_processing::LogicalQueryPlanPtr log_plan,
                               ClientPtr client) {
  TablePtr result;

  const std::string device_type =
      VariableManager::instance().getVariableValueString(
          "code_gen.cl_device_type");

  auto query_type = VariableManager::instance().getVariableValueString(
      "bench.exploration.querytype");

  auto exploration_max_pipe_exec_time_in_s =
      VariableManager::instance().getVariableValueInteger(
          "bench.exploration.max_pipe_exec_time_in_s");

  VariantIterator variant_iterator;
  ExplorationLogger explog(client, getOpenCLGlobalDevice(), "Full Exploration",
                           "[Training]", "[TrainingHeader]");

  assert(VariableManager::instance().getVariableValueString(
             "code_gen.exec_strategy") == "opencl");

  if (query_type == "projection" ||
      query_type == "grouped_aggregation_with_join" ||
      query_type == "aggregation_with_join") {
    /* temporarily remove strategy "parallel_global_atomic_single_pass" from
     * experiments */
    auto& dimension = variant_iterator.add(
        "code_gen.pipe_exec_strategy",
        {"serial_single_pass", "parallel_three_pass"}, 100);
    dimension.addChilds({"parallel_three_pass"}, /* temporarily remove parallel
                                                    global atomic strategy from
                                                    experiments */
                        "code_gen.projection.global_size_multiplier",
                        {"1", "8", "64", "256", "1024", "16384", "65536"});
  }

  if (query_type == "grouped_aggregation" ||
      query_type == "grouped_aggregation_with_join") {
    auto& dimension =
        variant_iterator.add("code_gen.opt.ocl_grouped_aggregation_strategy",
                             {"atomic_workgroup", "sequential", "atomic"}, 100);

    dimension.addChilds({"atomic_workgroup"},
                        "code_gen.opt.ocl_grouped_aggregation.atomic."
                        "workgroup.local_size",
                        {"16", "32", "64", "128", "256", "512", "1024"});

    dimension.addChilds(
        {"atomic_workgroup", "atomic"},
        "code_gen.opt.ocl_grouped_aggregation.global_size_multiplier",
        {"1", "8", "64", "256", "1024", "16384", "65536"});

    dimension.addChilds({"atomic_workgroup", "sequential", "atomic"},
                        "code_gen.opt.ocl_grouped_aggregation.hash_function",
                        {"murmur3", "multiply_shift"});

    dimension.addChilds({"atomic_workgroup", "sequential", "atomic"},
                        "code_gen.opt.ocl_grouped_aggregation_hashtable",
                        {"linear_probing", "cuckoo_hashing"});
  }

  variant_iterator.add("code_gen.memory_access", {"coalesced", "sequential"});
  variant_iterator.add("code_gen.opt.enable_predication", {"false", "true"});

  Variant standard_variant = getCurrentVariant(variant_iterator);

  if (variant_exploration_mode == "full_exploration") {
    auto start = getTimestamp();
    std::vector<VariantPerformance> variant_performance;

    if (variant_iterator.empty()) {
      std::cout << "VariantIterator has now variants, aborting "
                << "the full exploration!" << std::endl;
    }

    /* perform warmup queries, so data is always in-memory and loaded into
     * the data caches
     */
    auto warmup_query_count = 2u;
    std::cout << "Performing Warmup " << warmup_query_count << " Queries"
              << std::endl;
    for (auto i = 0u; i < warmup_query_count; ++i) {
      executeQueryPlanWithCompiler(log_plan, client, code_gen_type,
                                   &*variant_iterator.begin());
    }

    /* execute benchmark */
    auto total_number_of_variants =
        std::distance(variant_iterator.begin(), variant_iterator.end());
    std::cout << "Start Variant Exploration Benchmark testing "
              << total_number_of_variants << " variants" << std::endl;
    uint64_t current_variant_number = 1;
    for (const auto& variant : variant_iterator) {
      std::cout << "Execute Variant " << current_variant_number << "/"
                << total_number_of_variants << std::endl;
      current_variant_number++;
      print(client, variant);

      std::vector<VariantMeasurement> measurements;
      VariantExecutionStatistics statistics;
      double std_dev = 0;
      int counter = 0;
      do {
        /* fillmeasurements */
        for (auto i = 0u; i < 5u; ++i) {
          auto ret = executeQueryPlanWithCompiler(log_plan, client,
                                                  code_gen_type, &variant);
          result = ret.first;
          print(client, ret.second);

          measurements.push_back(ret.second);

          if (ret.second.total_pipeline_execution_time_in_s >
              exploration_max_pipe_exec_time_in_s) {
            break;
          }
        }

        std::sort(measurements.begin(), measurements.end(),
                  [](const VariantMeasurement& a,
                     const VariantMeasurement& b) -> bool {
                    return a.total_pipeline_execution_time_in_s <
                           b.total_pipeline_execution_time_in_s;
                  });

        statistics = VariantExecutionStatistics(measurements);
        std_dev = statistics.standard_deviation;
        // remove outliers
        while (measurements.size() > 2 && std_dev > 0.1) {
          measurements.pop_back();
          statistics = VariantExecutionStatistics(measurements);
          std_dev = statistics.standard_deviation;
        }
        // call fillmeasurements so often that 5 at least measurements are
        // available

        VariantPerformance vp(variant, statistics);
        variant_performance.push_back(vp);

        if (counter >= 5) {
          COGADB_FATAL_ERROR(
              "Standard deviation of the measurements is too high.", "");
        }
        counter++;
      } while (std_dev > 0.1);
    }

    std::sort(
        variant_performance.begin(), variant_performance.end(),
        [](const VariantPerformance& a, const VariantPerformance& b) -> bool {
          return a.second.mean < b.second.mean;
        });

    explog.logHeader();

    if (variant_performance.begin() != variant_performance.end()) {
      auto it = variant_performance.begin();
      const auto end = --variant_performance.end();
      explog.log(*it, "Fastest");
      fastest_variant = Variant((*it).first);
      while (++it != end) {
        explog.log(*it, "Variant");
      }
      explog.log(*it, "Slowest");
    }
    auto end = getTimestamp();

    std::cout << "FullExplorationRuntime: " << (end - start) / (1e9) << "s"
              << std::endl;
  } else if (variant_exploration_mode == "feature_wise_exploration") {
    /*******************************************************************************************/
    auto start = getTimestamp();
    /* perform warmup queries, so data is always in-memory and loaded into
     * the data caches
     */
    auto warmup_query_count = 2u;
    std::cout << "Performing Warmup " << warmup_query_count << " Queries"
              << std::endl;
    for (auto i = 0u; i < warmup_query_count; ++i) {
      executeQueryPlanWithCompiler(log_plan, client, code_gen_type,
                                   &*variant_iterator.begin());
    }

    ExplorationLogger explog(client, getOpenCLGlobalDevice(),
                             "Feature-wise Exploration", "[Training]",
                             "[TrainingHeader]");

    uint32_t max_iteration =
        VariableManager::instance().getVariableValueInteger(
            "code_gen.feature_wise_exploration.max_iteration_count");

    Variant best_variant_ever;
    double runtime_best_variant_ever = std::numeric_limits<double>::max();

    Variant predicted_best_variant =
        getStartVariantExloration(variant_iterator);
    VariantPerformance predicted_best_variant_performance;
    double last_variant_runtime = std::numeric_limits<double>::max();

    VariantConfigurator vc;
    uint32_t variant_counter = 0;

    for (auto iteration = 0u; iteration < max_iteration; ++iteration) {
      // if (iteration > 0) {
      vc(predicted_best_variant);
      //}

      /* core algortihm, search for best variant using a ceteris paribus
       * analysis */
      bool doRestart = false;
      double std_dev = 0;
      int restartCounter = 0, counter = 0;
      VariantExecutionStatistics statistics(0, 0, 0, 0, 0, 0);
      std::vector<VariantPerformance> variant_performance;

      // Feature
      auto flattened_dimensions = variant_iterator.getFlattenedDimensions();
      for (auto& dimension : flattened_dimensions) {
        std::vector<VariantPerformance> feature_variant_performance;
        // Variant
        for (const auto& value : dimension.second) {
          ++variant_counter;
          /* create variant to measure */
          auto variant = predicted_best_variant;
          variant[dimension.first] = value;
          /* measurements for current variant */
          std::vector<VariantMeasurement> measurements;

          // Run
          std::cout << "#######################################################"
                       "########"
                    << std::endl;
          std::cout << "Execute Variant: " << variant_counter << std::endl;
          //          std::cout << predicted_best_variant  << std::endl;

          /* perform measurements */
          do {
            for (auto i = 0u; i < 5; ++i) {
              auto ret = executeQueryPlanWithCompiler(log_plan, client,
                                                      code_gen_type, &variant);
              result = ret.first;
              print(client, ret.second);

              measurements.push_back(ret.second);

              if (ret.second.total_pipeline_execution_time_in_s >
                  exploration_max_pipe_exec_time_in_s) {
                break;
              }
            }
            doRestart = false;

            statistics = VariantExecutionStatistics(measurements);
            std_dev = statistics.standard_deviation;

            // if the measurements are of extremely poor quality, discard the
            // current results and rerun the experiments
            if (std_dev > 2) {
              if (restartCounter >= 3) {
                COGADB_FATAL_ERROR(
                    "Cannot obtain any reliable measurements from this device. "
                    "The standard deviation is extremely high. Canceling "
                    "Experiments.",
                    "");
              }
              measurements.clear();
              restartCounter++;
              continue;
            }

            std::sort(measurements.begin(), measurements.end(),
                      [](const VariantMeasurement& a,
                         const VariantMeasurement& b) -> bool {
                        return a.total_pipeline_execution_time_in_s <
                               b.total_pipeline_execution_time_in_s;
                      });

            double median;
            if (measurements.size() % 2 == 0) {
              median = (measurements[measurements.size() / 2 - 1]
                            .total_pipeline_execution_time_in_s +
                        measurements[measurements.size() / 2]
                            .total_pipeline_execution_time_in_s) /
                       2;
            } else {
              median = measurements[measurements.size() / 2]
                           .total_pipeline_execution_time_in_s;
            }

            std::cout << __FILE__ << ": " << __LINE__ << std::endl;
            std::cout << "measurements: ";
            for (auto& measurement : measurements) {
              std::cout << measurement.total_pipeline_execution_time_in_s
                        << ", ";
            }
            printf("   ");
            std::cout << "median: " << median << "curMax: "
                      << measurements.back().total_pipeline_execution_time_in_s
                      << std::endl;

            // remove outliers based on the difference to the median
            while ((measurements.size() > 2) &&
                   ((measurements.back().total_pipeline_execution_time_in_s /
                     median) >= OUTLIER_DIFF_THRESHOLD)) {
              printf("Removing element: %f ",
                     measurements.back().total_pipeline_execution_time_in_s);
              measurements.pop_back();
            }

            std::cout << "Measurement size: " << measurements.size() << " ";

            // if there are less than 5 measurements, make another run to get
            // enough measurements
            if (measurements.size() < 5) {
              doRestart = true;
              counter++;
            }

            statistics = VariantExecutionStatistics(measurements);
            std_dev = statistics.standard_deviation;

            std::cout << "restartCounter=" << restartCounter
                      << " counter=" << counter << " std_dev=" << std_dev
                      << " threshold=" << OUTLIER_DIFF_THRESHOLD << std::endl;

            if (counter >= 5) {
              COGADB_FATAL_ERROR(
                  "After 5 retries there are still not enough reliable "
                  "measurements. Canceling experiments.",
                  "");
            }
          } while (doRestart);

          VariantPerformance vp(variant, statistics);
          feature_variant_performance.push_back(vp);
          counter = 0;
          std::cout << "#######################################################"
                       "########"
                    << std::endl;
        }

        /* determine best variant for the current dimension */
        std::sort(feature_variant_performance.begin(),
                  feature_variant_performance.end(),
                  [](const VariantPerformance& a, const VariantPerformance& b)
                      -> bool { return a.second.mean < b.second.mean; });

        if (feature_variant_performance.begin() !=
            feature_variant_performance.end()) {
          auto it = feature_variant_performance.begin();
          const auto end = --feature_variant_performance.end();

          explog.log(*it, "Fast");
          variant_performance.push_back(*it);
          vc(it->first);

          //          VariantPerformance fastest_variant_performance=*it;
          predicted_best_variant = (*it).first;
          predicted_best_variant_performance = *it;

          for (const auto& feature : predicted_best_variant) {
            client->getOutputStream() << "Iteration " << iteration
                                      << " Dimension: " << feature.first << " "
                                      << "Best Value: " << feature.second
                                      << std::endl;
          }
          std::cout << "[Fastest Variant]: "
                    << predicted_best_variant_performance.second.median << "s"
                    << std::endl;

          while (++it != end) {
            explog.log(*it, "Variant");
          }
          explog.log(*it, "Slow");
        }
      } /* end feature-wise exploration (core loop) */

      std::cout << "[Best Variant Iteration " << iteration << "]" << std::endl;
      for (const auto& feature : predicted_best_variant) {
        client->getOutputStream() << "Iteration " << iteration
                                  << " Dimension: " << feature.first << " "
                                  << "Best Value: " << feature.second
                                  << std::endl;
      }

      //      //predicted_best_variant.clear();
      //      for (const auto& vp : variant_performance) {
      //        for (const auto& feature : vp.first) {
      //          //predicted_best_variant.insert(feature);
      //          client->getOutputStream() << "Iteration " << iteration
      //                                    << " Dimension: " << feature.first
      //                                    << " "
      //                                    << "Best Value: " << feature.second
      //                                    << std::endl;
      //        }
      //      }
      //      predicted_best_variant_performance

      counter = 0;
      /* check whether variant performs significantly faster or worse,
       * store result in fastest_variant
       */
      //      statistics = VariantExecutionStatistics(measurements);
      //      VariantPerformance vp(predicted_best_variant, statistics);
      explog.log(predicted_best_variant_performance,
                 "Iteration" + std::to_string(iteration));

      if (0.9 * last_variant_runtime <=
              predicted_best_variant_performance.second.mean &&
          1.1 * last_variant_runtime >=
              predicted_best_variant_performance.second.mean) {
        break;
      } else {
        last_variant_runtime = predicted_best_variant_performance.second.mean;
      }
    } /* end of iteration */

    fastest_variant = Variant(predicted_best_variant);

    static std::map<std::string, std::map<std::string, uint32_t>>
        variants_over_queries;

    for (const auto& feature : predicted_best_variant) {
      variants_over_queries[feature.first][feature.second]++;
    }

    std::string result_filename =
        device_type + "_feature_wise_best_variant.coga";
    std::ofstream result_config(result_filename);

    for (const auto& dim : variants_over_queries) {
      std::string value = "";
      uint32_t highest = 0;

      for (const auto& val : dim.second) {
        if (val.second > highest) {
          highest = val.second;
          value = val.first;
        }
      }

      result_config << "set " << dim.first << "=" << value << std::endl;
    }

    auto end = getTimestamp();

    std::cout << "FeatureWiseExplorationRuntime: " << (end - start) / (1e9)
              << "s" << std::endl;
    std::cout << "FeatureWiseExecutedVariants: " << variant_counter
              << std::endl;

    /*******************************************************************************************/
  } else if (variant_exploration_mode == "genetic") {
    auto start = getTimestamp();
    /* perform warmup queries, so data is always in-memory and loaded into
     * the data caches
     */
    auto warmup_query_count = 2u;
    std::cout << "Performing Warmup " << warmup_query_count << " Queries"
              << std::endl;
    for (auto i = 0u; i < warmup_query_count; ++i) {
      executeQueryPlanWithCompiler(log_plan, client, code_gen_type,
                                   &*variant_iterator.begin());
    }

    genetic_exploration(variant_iterator, log_plan, code_gen_type, client);

    auto end = getTimestamp();

    std::cout << "GeneticExplorationRuntime: " << (end - start) / (1e9) << "s"
              << std::endl;
  }
#ifdef PERSEUS_FOUND
  else if (variant_exploration_mode == "perseus") {
    result = executeQueryPlanWithPerseus(log_plan, client, code_gen_type,
                                         variant_iterator);
  }
#endif
  else {
    COGADB_FATAL_ERROR("Unknown or unsupported variant exploration mode: "
                           << variant_exploration_mode,
                       "");
  }

  // reset the configuration
  VariantConfigurator vc;
  vc(standard_variant);

  return result;
}

const TablePtr executeQueryPlan(query_processing::LogicalQueryPlanPtr log_plan,
                                ClientPtr client) {
  std::ostream& out = client->getOutputStream();

#ifdef COGADB_MEASURE_ENERGY_CONSUMPTION
  PCM* pcm = PCM::getInstance();

  SystemCounterState systemState_before;
  SystemCounterState systemState_after;

  systemState_before = pcm->getSystemCounterState();
  Timestamp begin = getTimestamp();
#endif

  TablePtr result;
  const std::string query_exec_policy =
      VariableManager::instance().getVariableValueString(
          "query_execution_policy");

  if (query_exec_policy == "compiled") {
    //    Timestamp begin_total_time = getTimestamp();
    CodeGeneratorType code_gen_type;
    std::string code_gen_name =
        VariableManager::instance().getVariableValueString(
            "default_code_generator");

    convertToCodeGenerator(code_gen_name, code_gen_type);

    const std::string variant_exploration_mode =
        VariableManager::instance().getVariableValueString(
            "code_gen.variant_exploration_mode");
    if (variant_exploration_mode == "no_exploration" ||
        VariableManager::instance().getVariableValueString(
            "code_gen.exec_strategy") != "opencl") {
      std::pair<TablePtr, VariantMeasurement> ret =
          executeQueryPlanWithCompiler(log_plan, client, code_gen_type,
                                       &fastest_variant);
      result = ret.first;
      print(client, ret.second);
    } else {
      result = exploreVariants(variant_exploration_mode, code_gen_type,
                               log_plan, client);
    }
  } else if (query_exec_policy == "interpreted") {
    CoGaDB::query_processing::PhysicalQueryPlanPtr result_plan =
        CoGaDB::query_processing::optimize_and_execute("", *log_plan, client);
    assert(result_plan != NULL);
    result = result_plan->getResult();
  } else {
    COGADB_FATAL_ERROR(
        "Invalid query execution policy: '" << query_exec_policy << "'", "");
  }
  if (result) {
    TablePtr materialized_table = getTablebyName(result->getName());
    if (!materialized_table || materialized_table != result) {
      // is result, so set empty string as table name
      result->setName("");
    }
  }

#ifdef COGADB_MEASURE_ENERGY_CONSUMPTION
  Timestamp end = getTimestamp();
  systemState_after = pcm->getSystemCounterState();
#endif

#ifdef COGADB_MEASURE_ENERGY_CONSUMPTION
  double consumed_joules_processor =
      getConsumedJoules(systemState_before, systemState_after);
  double consumed_joules_dram =
      getDRAMConsumedJoules(systemState_before, systemState_after);
  double total_joules_for_query =
      consumed_joules_processor + consumed_joules_dram;

  out << "Consumed Joules (CPU): " << consumed_joules_processor << std::endl;
  out << "Consumed Joules (DRAM): " << consumed_joules_dram << std::endl;
  out << "Consumed Joules (Total): " << total_joules_for_query << std::endl;
  out << "Energy Product (Joule*seconds): "
      << total_joules_for_query * double(end - begin) / (1000 * 1000 * 1000)
      << std::endl;
#endif

  return result;
}

bool printResult(TablePtr table, ClientPtr client,
                 double exec_time_in_milliseconds) {
  std::ostream& out = client->getOutputStream();

  if (!table) {
    out << "Error: Invalid Result Table (TablePtr is NULL)" << std::endl;
    return false;
  }

  bool print_query_result =
      VariableManager::instance().getVariableValueBoolean("print_query_result");
  if (print_query_result)
    out << table->toString();
  else
    out << "[result size: " << table->getNumberofRows() << " rows]";

  if (VariableManager::instance().getVariableValueString(
          "result_output_format") == "table") {
    std::stringstream time_str;
    time_str.precision(5);
    time_str << exec_time_in_milliseconds;
    out << std::endl
        << "Execution Time: " << time_str.str() << " ms" << std::endl;
  }

  return true;
}

}  // end namespace CoGaDB
