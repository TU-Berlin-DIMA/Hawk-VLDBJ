
#include <core/data_dictionary.hpp>
#include <core/processor_data_cache.hpp>
#include <query_processing/join_operator.hpp>
#include <util/hardware_detector.hpp>

#include <query_compilation/code_generator.hpp>
#include <query_compilation/predicate_expression.hpp>
#include <query_compilation/query_context.hpp>

namespace CoGaDB {

namespace query_processing {

// Map_Init_Function
// init_function_Selection_operator=physical_operator::map_init_function_Selection_operator;

namespace logical_operator {

const hype::Tuple Logical_Join::getFeatureVector() const {
  hype::Tuple t;
  if (this->left_) {  // if left child is valid (has to be by convention!), add
                      // input data size
    // if we already know the correct input data size, because the child node
    // was already executed
    // during query chopping, we use the real cardinality, other wise we call
    // the estimator
    if (this->left_->getPhysicalOperator()) {
      t.push_back(this->left_->getPhysicalOperator()
                      ->getResultSize());  // ->result_size_;
    } else {
      t.push_back(this->left_->getOutputResultSize());
    }
    if (this->right_) {  // if right child is valid (not null), add input data
                         // size for it as well
      if (this->right_->getPhysicalOperator()) {
        t.push_back(this->right_->getPhysicalOperator()
                        ->getResultSize());  // ->result_size_;
      } else {
        t.push_back(this->right_->getOutputResultSize());
      }
    }
  } else {
    HYPE_FATAL_ERROR("Invalid Left Child!", std::cout);
  }
  if (useSelectivityEstimation())
    t.push_back(this->getSelectivity());  // add selectivity of this operation,
                                          // when use_selectivity_estimation_ is
                                          // true
#ifdef HYPE_INCLUDE_INPUT_DATA_LOCALITY_IN_FEATURE_VECTOR
  // dirty workaround! We should make isInputDataCachedInGPU() a const member
  // function!
  //                t.push_back(const_cast<Node*>(this)->isInputDataCachedInGPU());
  //                t.push_back(this->isInputDataCachedInGPU_internal().second);
  t.push_back(this->isInputDataCachedInGPU_internal().first);
#endif
  return t;
}

typedef hype::queryprocessing::TypedOperator<TablePtr> PhysicalTableOperator;
typedef boost::shared_ptr<PhysicalTableOperator> PhysicalTableOperatorPtr;

typedef hype::queryprocessing::UnaryOperator<TablePtr, TablePtr>
    PhysicalTableUnaryOperator;
typedef boost::shared_ptr<PhysicalTableUnaryOperator>
    PhysicalTableUnaryOperatorPtr;

typedef hype::queryprocessing::BinaryOperator<TablePtr, TablePtr, TablePtr>
    PhysicalTableBinaryOperator;
typedef boost::shared_ptr<PhysicalTableBinaryOperator>
    PhysicalTableBinaryOperatorPtr;

std::pair<bool, double> Logical_Join::isInputDataCachedInGPU_internal() const {
  double number_of_cached_access_strutures = 0;
  hype::ProcessingDeviceMemoryID mem_id = getMemoryIDForDeviceID(0);
  bool col1_cached = false, col2_cached = false;
  {
    if (this->left_ && this->right_ && this->getLeft()->getPhysicalOperator() &&
        this->getRight()->getPhysicalOperator()) {
      PhysicalTableOperatorPtr left_phy_op =
          boost::dynamic_pointer_cast<PhysicalTableOperator>(
              this->getLeft()->getPhysicalOperator());
      PhysicalTableOperatorPtr right_phy_op =
          boost::dynamic_pointer_cast<PhysicalTableOperator>(
              this->getRight()->getPhysicalOperator());

      assert(left_phy_op != NULL);
      assert(right_phy_op != NULL);

      TablePtr result_table_left = left_phy_op->getResult();
      TablePtr result_table_right = right_phy_op->getResult();
      if (result_table_left && result_table_right) {
        ColumnPtr col1 =
            result_table_left->getColumnbyName(this->join_column1_name_);
        ColumnPtr col2 =
            result_table_right->getColumnbyName(this->join_column2_name_);
        if (col1 && col2) {
          DataCache& gpu_cache =
              DataCacheManager::instance().getDataCache(mem_id);
          bool col1_cached = gpu_cache.isCached(col1);
          bool col2_cached = gpu_cache.isCached(col2);
          /* Check whether columns are stored in GPU memory,
           even if they are not listed in the cache. */
          if (!col1_cached) {
            if (col1->getMemoryID() == mem_id) {
              col1_cached = true;
            }
          }
          if (!col2_cached) {
            if (col2->getMemoryID() == mem_id) {
              col2_cached = true;
            }
          }
          /* Printout decisions to user */
          (*this->out) << "Column '" << col1->getName() << "' from table '"
                       << result_table_left->getName()
                       << "' is cached: " << col1_cached << std::endl;
          (*this->out) << "Column '" << col2->getName() << "' from table '"
                       << result_table_right->getName()
                       << "' is cached: " << col2_cached << std::endl;
          if ((col1_cached || result_table_left->getNumberofRows() < 1000) &&
              (col2_cached || result_table_right->getNumberofRows() < 1000)) {
            (*this->out) << "Operator '" << this->toString(true)
                         << "' determined its input data is cached!"
                         << std::endl;
            size_t num_cached_columns = 0;
            if (col1_cached) num_cached_columns++;
            if (col2_cached) num_cached_columns++;
            return std::make_pair(true, double(num_cached_columns));
          } else {
            // make an exception if we have exactly one CPU
            // only selection operators as childs (apply the
            // generic tests below)
            // this catches the case where greater equal predicates
            // are evaluated on dictionary compressed columns,
            // which we use for strings
            if (!col1_cached && col2_cached &&
                left_->getDeviceConstraint() == hype::CPU_ONLY &&
                left_->getOperationName() == "COMPLEX_SELECTION") {
              // do nothing and run the other tests below
            } else if (col1_cached && !col2_cached &&
                       right_->getDeviceConstraint() == hype::CPU_ONLY &&
                       right_->getOperationName() == "COMPLEX_SELECTION") {
              // do nothing and run the other tests below
            } else {
              size_t num_cached_columns = 0;
              if (col1_cached) num_cached_columns++;
              if (col2_cached) num_cached_columns++;
              return std::make_pair(false, double(num_cached_columns));
            }
          }
        }
      }
    }
    AttributeReferencePtr attr =
        getAttributeFromColumnIdentifier(this->join_column1_name_);
    if (!attr) {
      COGADB_WARNING("Join Column "
                         << this->join_column1_name_
                         << " not found! I will assume it is not cached!",
                     "");
      return std::make_pair(false, double(0));
    }
    ColumnPtr col = attr->getColumn();
    assert(col != NULL);
    col1_cached =
        DataCacheManager::instance().getDataCache(mem_id).isCached(col);
  }
  {
    AttributeReferencePtr attr =
        getAttributeFromColumnIdentifier(this->join_column2_name_);
    if (!attr) {
      COGADB_WARNING("Join Column "
                         << this->join_column2_name_
                         << " not found! I will assume it is not cached!",
                     "");
      return std::make_pair(false, double(0));
    }

    ColumnPtr col = attr->getColumn();
    assert(col != NULL);
    col2_cached =
        DataCacheManager::instance().getDataCache(mem_id).isCached(col);
  }
  if (col1_cached) number_of_cached_access_strutures++;
  if (col2_cached) number_of_cached_access_strutures++;

  if (col1_cached && col2_cached) {
    if (this->getLeft()->getOperationName() != "SCAN" &&
        this->getRight()->getOperationName() != "SCAN") {
      if (this->getLeft()->getPhysicalOperator() &&
          this->getLeft()
                  ->getPhysicalOperator()
                  ->getDeviceSpecification()
                  .getDeviceType() == hype::GPU &&
          this->getRight()->getPhysicalOperator() &&
          this->getRight()
                  ->getPhysicalOperator()
                  ->getDeviceSpecification()
                  .getDeviceType() == hype::GPU) {
        return std::make_pair(true, number_of_cached_access_strutures + 2);
      }
      if (left_->getDeviceConstraint().getDeviceTypeConstraint() ==
              hype::GPU_ONLY &&
          right_->getDeviceConstraint().getDeviceTypeConstraint() ==
              hype::GPU_ONLY) {
        return std::make_pair(true, number_of_cached_access_strutures + 2);
      }
    } else if (this->getLeft()->getOperationName() != "SCAN" &&
               this->getRight()->getOperationName() == "SCAN") {
      if (this->getLeft()->getPhysicalOperator() &&
          this->getLeft()
                  ->getPhysicalOperator()
                  ->getDeviceSpecification()
                  .getDeviceType() == hype::GPU) {
        return std::make_pair(true, number_of_cached_access_strutures + 1);
      }
      if (left_->getDeviceConstraint().getDeviceTypeConstraint() ==
          hype::GPU_ONLY) {
        return std::make_pair(true, number_of_cached_access_strutures + 1);
      }
    } else if (this->getLeft()->getOperationName() == "SCAN" &&
               this->getRight()->getOperationName() != "SCAN") {
      if (this->getRight()->getPhysicalOperator() &&
          this->getRight()
                  ->getPhysicalOperator()
                  ->getDeviceSpecification()
                  .getDeviceType() == hype::GPU) {
        return std::make_pair(true, number_of_cached_access_strutures + 1);
      }
      if (right_->getDeviceConstraint().getDeviceTypeConstraint() ==
          hype::GPU_ONLY) {
        return std::make_pair(true, number_of_cached_access_strutures + 1);
      }
    }
    return std::make_pair(true, number_of_cached_access_strutures);
  }
  // not cached both input columns, make flag false
  return std::make_pair(false, number_of_cached_access_strutures);
}

Logical_Join::Logical_Join(const std::string& join_column1_name,
                           const std::string& join_column2_name,
                           const JoinType& join_type,
                           hype::DeviceConstraint dev_constr)
    : TypedNode_Impl<TablePtr,
                     physical_operator::map_init_function_join_operator>(
          false, dev_constr),
      join_column1_name_(join_column1_name),
      join_column2_name_(join_column2_name),
      join_type_(join_type),
      consume_comes_from_left_sub_tree_(true) {
  join_column1_name_ =
      convertToFullyQualifiedNameIfRequired(join_column1_name_);
  join_column2_name_ =
      convertToFullyQualifiedNameIfRequired(join_column2_name_);
}

unsigned int Logical_Join::getOutputResultSize() const { return 10; }

double Logical_Join::getCalculatedSelectivity() const { return 1; }

std::string Logical_Join::getOperationName() const {
  return util::getName(this->join_type_);
}

std::string Logical_Join::toString(bool verbose) const {
  std::string result = util::getName(this->join_type_);  //"JOIN";
  if (verbose) {
    result += " (";
    result += join_column1_name_;
    result += "=";
    result += join_column2_name_;
    result += ")";
  }
  return result;
}
const std::string& Logical_Join::getLeftJoinColumnName() {
  return join_column1_name_;
}

const std::string& Logical_Join::getRightJoinColumnName() {
  return join_column2_name_;
}

const JoinType Logical_Join::getJoinType() const { return join_type_; }

bool Logical_Join::isInputDataCachedInGPU() {
  return isInputDataCachedInGPU_internal().first;
}

void Logical_Join::produce_impl(CodeGeneratorPtr code_gen,
                                QueryContextPtr context) {
  /* The build of the hash table is a pipeline breaker. Thus, we
   * generate a new code generator and a new query context */
  ProjectionParam param;
  CodeGeneratorPtr code_gen_build_phase =
      createCodeGenerator(code_gen->getCodeGeneratorType(), param);
  QueryContextPtr context_build_phase =
      createQueryContext(BUILD_HASH_TABLE_PIPELINE);
  /* we first call produce on the left child, so if we come
   * back calling the consume function, we need
   * to generate code to build the hash table in the consume call
   */
  consume_comes_from_left_sub_tree_ = true;
  /* add the attributes accessed by this operator to the list in
   * the query context */
  context->addAccessedColumn(this->join_column1_name_);
  context->addAccessedColumn(this->join_column2_name_);

  context_build_phase->fetchInformationFromParentContext(context);
  /*
   * An empty projection list signales that we need all attributes from the
   * table.
   * Thus we check whether the list is empty, and if not, we add our
   * build site attribute to it.
   */
  if (!context_build_phase->getProjectionList().empty())
    context_build_phase->addColumnToProjectionList(this->join_column1_name_);
  /* create the build pipeline */
  left_->produce(code_gen_build_phase, context_build_phase);
  //                code_gen_build_phase->print();
  //                context_build_phase->print(*this->out);
  PipelinePtr build_pipeline = code_gen_build_phase->compile();
  if (!build_pipeline) {
    COGADB_FATAL_ERROR("Compiling code for hash table build failed!", "");
  }
  TablePtr result = execute(build_pipeline, context_build_phase);
  if (!result) {
    COGADB_FATAL_ERROR(
        "Execution of compiled code for hash table build failed!", "");
  }
  //                context->updateStatistics(build_pipeline);
  context->updateStatistics(context_build_phase);
  storeResultTableAttributes(code_gen, context, result);
  /* now we call produce on the right child, so if we come
   * back calling the consume function,  we need
   * to generate code to probe the hash table in the consume call
   */
  consume_comes_from_left_sub_tree_ = false;
  //                result->print();
  right_->produce(code_gen, context);
  //                code_gen->print();
}

void Logical_Join::consume_impl(CodeGeneratorPtr code_gen,
                                QueryContextPtr context) {
  //  if (context->getPipelineType() == BUILD_HASH_TABLE_PIPELINE) {
  if (this->consume_comes_from_left_sub_tree_) {
    /* build phase */
    AttributeReferencePtr left_join_attr =
        code_gen->getScannedAttributeByName(this->join_column1_name_);

    if (!left_join_attr) {
      COGADB_FATAL_ERROR("Could not retrieve Attribute Reference of attribute '"
                             << this->join_column1_name_
                             << "' for build side of hash join "
                                "for operator "
                             << this->toString(true) << "!",
                         "");
    }
    if (!code_gen->consumeBuildHashTable(*left_join_attr)) {
      COGADB_FATAL_ERROR("Failed to generate build pipeline for join!", "");
    }
    /* this is a pipeline breaker, so we must not call consume of the parent
     * operator! */

  } else {
    /* probe phase */
    AttributeReferencePtr left_join_attr =
        context->getAttributeFromOtherPipelineByName(this->join_column1_name_);
    if (!left_join_attr) {
      COGADB_FATAL_ERROR("Could not retrieve Attribute Reference to column "
                             << this->join_column1_name_
                             << " for build side of hash join "
                                "for operator "
                             << this->toString(true) << "!",
                         "");
    }

    AttributeReferencePtr right_join_attr =
        code_gen->getScannedAttributeByName(this->join_column2_name_);
    if (!right_join_attr) {
      COGADB_FATAL_ERROR("Could not retrieve Attribute "
                             << "Reference to column "
                             << this->join_column2_name_
                             << " for probe side of hash join!",
                         "");
    }

    if (!code_gen->consumeProbeHashTable(*left_join_attr, *right_join_attr)) {
      COGADB_FATAL_ERROR("Failed to generated probe pipeline for join!", "");
    }
    /* we can pipeline the probe, so consume the parent operator */
    if (this->parent_) {
      this->parent_->consume(code_gen, context);
    }
  }
}

}  // end namespace logical_operator

namespace physical_operator {

TypedOperatorPtr create_Join_Operator(TypedLogicalNode& logical_node,
                                      const hype::SchedulingDecision& sched_dec,
                                      TypedOperatorPtr left_child,
                                      TypedOperatorPtr right_child,
                                      const JoinAlgorithm& join_alg) {
  logical_operator::Logical_Join& log_join_ref =
      static_cast<logical_operator::Logical_Join&>(logical_node);
  if (!quiet && verbose && debug)
    std::cout << "create CPU_NestedLoopJoin_Operator!" << std::endl;
  if (!left_child) {
    std::cout << "Error! File: " << __FILE__ << " Line: " << __LINE__
              << std::endl;
    exit(-1);
  }

  hype::ProcessingDeviceID id =
      sched_dec.getDeviceSpecification().getProcessingDeviceID();
  ProcessorSpecification proc_spec(id);
  JoinParam param(proc_spec, join_alg, log_join_ref.getJoinType());

  assert(right_child != NULL);  // binary operator
  return TypedOperatorPtr(new Join_Operator(
      sched_dec, left_child, right_child, log_join_ref.getLeftJoinColumnName(),
      log_join_ref.getRightJoinColumnName(), param));
}

TypedOperatorPtr create_NestedLoopJoin_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  return create_Join_Operator(logical_node, sched_dec, left_child, right_child,
                              NESTED_LOOP_JOIN);
}

TypedOperatorPtr create_SortMergeJoin_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  return create_Join_Operator(logical_node, sched_dec, left_child, right_child,
                              SORT_MERGE_JOIN);
}

TypedOperatorPtr create_HashJoin_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  return create_Join_Operator(logical_node, sched_dec, left_child, right_child,
                              HASH_JOIN);
}

//            TypedOperatorPtr
//            create_CPU_Parallel_HashJoin_Operator(TypedLogicalNode&
//            logical_node, const hype::SchedulingDecision& sched_dec,
//            TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
//                return create_Join_Operator(logical_node, sched_dec,
//                left_child, right_child, HASH_JOIN);
//            }

TypedOperatorPtr create_RadixJoin_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  return create_Join_Operator(logical_node, sched_dec, left_child, right_child,
                              RADIX_JOIN);
}

TypedOperatorPtr create_IndexNestedLoop_Join_Operator(
    TypedLogicalNode& logical_node, const hype::SchedulingDecision& sched_dec,
    TypedOperatorPtr left_child, TypedOperatorPtr right_child) {
  return create_Join_Operator(logical_node, sched_dec, left_child, right_child,
                              INDEX_NESTED_LOOP_JOIN);
}

Physical_Operator_Map_Ptr map_init_function_join_operator() {
  // std::cout << sd.getNameofChoosenAlgorithm() << std::endl;
  Physical_Operator_Map map;
  if (!quiet)
    std::cout << "calling map init function for JOIN operator!" << std::endl;
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_NestedLoopJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");
// stemod::Scheduler::instance().addAlgorithm("JOIN","CPU_SortMergeJoin_Algorithm",stemod::CPU,"Least
// Squares 2D","Periodic Recomputation");
// hype::Scheduler::instance().addAlgorithm("JOIN","CPU_HashJoin_Algorithm",hype::CPU,"Least
// Squares 2D","Periodic Recomputation");

#ifdef COGADB_USE_KNN_REGRESSION_LEARNER

  hype::AlgorithmSpecification join_alg_spec_cpu_hashjoin(
      "CPU_HashJoin_Algorithm", "JOIN", hype::KNN_Regression, hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_gpu_sort_merge_join(
      "GPU_SortMergeJoin_Algorithm", "JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_gpu_inlj_join(
      "GPU_IndexNestedLoopJoin_Algorithm", "JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_left_semi_join(
      "CPU_LEFT_SEMI_JOIN", "LEFT_SEMI_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_right_semi_join(
      "CPU_RIGHT_SEMI_JOIN", "RIGHT_SEMI_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_gpu_left_semi_join(
      "GPU_LEFT_SEMI_JOIN", "LEFT_SEMI_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_gpu_right_semi_join(
      "GPU_RIGHT_SEMI_JOIN", "RIGHT_SEMI_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_left_anti_semi_join(
      "CPU_LEFT_ANTI_SEMI_JOIN", "LEFT_ANTI_SEMI_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_right_anti_semi_join(
      "CPU_RIGHT_ANTI_SEMI_JOIN", "RIGHT_ANTI_SEMI_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_left_outer_join(
      "CPU_LEFT_OUTER_JOIN", "LEFT_OUTER_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_right_outer_join(
      "CPU_RIGHT_OUTER_JOIN", "RIGHT_OUTER_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_full_outer_join(
      "CPU_FULL_OUTER_JOIN", "FULL_OUTER_JOIN", hype::KNN_Regression,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_fetch_join(
      "CPU_GATHER_JOIN", "GATHER_JOIN", hype::KNN_Regression, hype::Periodic);

#else
  hype::AlgorithmSpecification join_alg_spec_cpu_nlj(
      "CPU_NestedLoopJoin_Algorithm", "JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_hashjoin(
      "CPU_HashJoin_Algorithm", "JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_parallel_hashjoin(
      "CPU_Parallel_HashJoin_Algorithm", "JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_radixjoin(
      "CPU_RadixJoin_Algorithm", "JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_gpu_sort_merge_join(
      "GPU_SortMergeJoin_Algorithm", "JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_gpu_inlj_join(
      "GPU_IndexNestedLoopJoin_Algorithm", "JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);

  hype::AlgorithmSpecification join_alg_spec_cpu_fetch_join(
      "CPU_FETCH_JOIN", "FETCH_JOIN", hype::Multilinear_Fitting_2D,
      hype::Periodic);
#endif
  // addAlgorithmSpecificationToHardware();

  const DeviceSpecifications& dev_specs =
      HardwareDetector::instance().getDeviceSpecifications();

  for (unsigned int i = 0; i < dev_specs.size(); ++i) {
    if (dev_specs[i].getDeviceType() == hype::CPU) {
#ifdef ENABLE_CPU_NESTED_LOOP_JOIN
      hype::Scheduler::instance().addAlgorithm(join_alg_spec_cpu_nlj,
                                               dev_specs[i]);
#endif
      hype::Scheduler::instance().addAlgorithm(join_alg_spec_cpu_hashjoin,
                                               dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(join_alg_spec_cpu_left_semi_join,
                                               dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(
          join_alg_spec_cpu_right_semi_join, dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(
          join_alg_spec_cpu_left_anti_semi_join, dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(
          join_alg_spec_cpu_right_anti_semi_join, dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(
          join_alg_spec_cpu_left_outer_join, dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(
          join_alg_spec_cpu_right_outer_join, dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(
          join_alg_spec_cpu_full_outer_join, dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(join_alg_spec_cpu_fetch_join,
                                               dev_specs[i]);

      // Radix Join has still some stability issues
      // hype::Scheduler::instance().addAlgorithm(join_alg_spec_cpu_radixjoin,
      // dev_specs[i]);
      // hype::Scheduler::instance().addAlgorithm(join_alg_spec_cpu_parallel_hashjoin,
      // dev_specs[i]);
    } else if (dev_specs[i].getDeviceType() == hype::GPU) {
      //		       hype::Scheduler::instance().addAlgorithm(join_alg_spec_gpu_sort_merge_join,dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(join_alg_spec_gpu_inlj_join,
                                               dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(join_alg_spec_gpu_left_semi_join,
                                               dev_specs[i]);
      hype::Scheduler::instance().addAlgorithm(
          join_alg_spec_gpu_right_semi_join, dev_specs[i]);

      //#ifdef ENABLE_GPU_ACCELERATION
      //     #ifdef ENABLE_GPU_JOIN
      //                        hype::Scheduler::instance().addAlgorithm(join_alg_spec_gpu_sort_merge_join,dev_specs[i]);
      //     #endif
      //#endif
    }
  }

  // stemod::Scheduler::instance().addAlgorithm("SELECTION","GPU_Selection_Algorithm",stemod::GPU,"Least
  // Squares 1D","Periodic Recomputation");
  map["CPU_NestedLoopJoin_Algorithm"] = create_NestedLoopJoin_Operator;
  // map["CPU_SortMergeJoin_Algorithm"]=create_CPU_SortMergeJoin_Operator;
  map["GPU_SortMergeJoin_Algorithm"] = create_SortMergeJoin_Operator;
  map["CPU_HashJoin_Algorithm"] = create_HashJoin_Operator;
  //                map["CPU_RadixJoin_Algorithm"] = create_RadixJoin_Operator;
  map["GPU_IndexNestedLoopJoin_Algorithm"] =
      create_IndexNestedLoop_Join_Operator;

  map["CPU_LEFT_SEMI_JOIN"] = create_HashJoin_Operator;
  map["CPU_RIGHT_SEMI_JOIN"] = create_HashJoin_Operator;
  map["GPU_LEFT_SEMI_JOIN"] = create_HashJoin_Operator;
  map["GPU_RIGHT_SEMI_JOIN"] = create_HashJoin_Operator;
  map["CPU_LEFT_ANTI_SEMI_JOIN"] = create_HashJoin_Operator;
  map["CPU_RIGHT_ANTI_SEMI_JOIN"] = create_HashJoin_Operator;
  map["CPU_LEFT_OUTER_JOIN"] = create_HashJoin_Operator;
  map["CPU_RIGHT_OUTER_JOIN"] = create_HashJoin_Operator;
  map["CPU_FULL_OUTER_JOIN"] = create_HashJoin_Operator;
  map["CPU_GATHER_JOIN"] = create_HashJoin_Operator;

  //                map["CPU_Parallel_HashJoin_Algorithm"] =
  //                create_CPU_Parallel_HashJoin_Operator;
  // map["GPU_Selection_Algorithm"]=create_GPU_Selection_Operator;
  return Physical_Operator_Map_Ptr(new Physical_Operator_Map(map));
}

}  // end namespace physical_operator

}  // end namespace query_processing

}  // end namespace CogaDB
