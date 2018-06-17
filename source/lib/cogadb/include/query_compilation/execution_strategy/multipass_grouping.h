/*
 * author: henning funke
 * date: 29.08.2016
 */

#ifndef OCL_MULTIPASS_GROUPING_H
#define OCL_MULTIPASS_GROUPING_H
#include <core/global_definitions.hpp>
#include <query_compilation/code_generators/ocl_code_generator_utils.hpp>
#include <query_compilation/execution_strategy/ocl.hpp>

namespace CoGaDB {

  std::string patchGroupAggregateKernelCode(std::string kernel_code);

  std::string getCodeCreateGroupAggregateStructures(
      std::vector<AttributeReference> groupingAttributes,
      std::vector<AttributeReference> aggregationAttributes,
      std::string num_elements);

  std::string getCodeMultipassGroupAggregate(
      std::vector<AttributeReference> groupingAttributes,
      std::vector<AttributeReference> aggregationAttributes,
      std::string input_size);

  std::string getCodeCreateGroupAggregateResult(
      std::vector<AttributeReference> groupingAttributes,
      std::vector<AttributeReference> aggregationAttributes);

  std::string getCodeCleanupGroupAggregateStructures(
      std::vector<AttributeReference> groupingAttributes,
      std::vector<AttributeReference> aggregationAttributes);

  std::string patchGroupAggregateHostCode(
      std::string host_code,
      std::vector<AttributeReference> aggregationAttributes,
      std::vector<AttributeReference> computedAttributes);

  std::string getCodeProjectGroupAggregateData();

}  // end namespace CoGaDB

#endif
