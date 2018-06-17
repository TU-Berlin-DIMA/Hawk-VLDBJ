/*
 * author: henning funke
 * date: 26.07.2016
 */

#ifndef OCL_WORKGROUP_UTILS_HPP
#define OCL_WORKGROUP_UTILS_HPP

#include <map>
#include <string>

namespace CoGaDB {

  enum ReduceDirection { UPSWEEP, DOWNSWEEP };
  enum ReduceImplementation { VAR1, VAR2, BUILTIN, WARP };

  typedef struct {
    std::string global_init;
    std::string kernel_init;
    std::string local_init;
    std::string computation;
  } WorkgroupCodeBlock;

  std::string getGroupVariableIndex();

  std::string getGroupResultOffset();

  bool isInclusiveImplementation();

  std::string getCustomScanCode(std::string upsweep_op, std::string clear_op,
                                std::string downsweep_op);

  std::string getSegmentedScan(std::map<std::string, std::string> aggVars,
                               std::string data_access_suffix);

  std::string getSegmentedScanWorkgroup(
      std::map<std::string, std::string> aggVars,
      std::string data_access_suffix);

  std::string getSegmentedReductionWarpInclusive(
      std::string head_var, std::map<std::string, std::string> aggVars,
      std::string suffix);

  WorkgroupCodeBlock getReduceCode(ReduceImplementation imp, std::string type,
                                   std::string input_var,
                                   std::string output_var, int workgroup_size,
                                   std::string larr = "temp");

  WorkgroupCodeBlock getReduceCodeGenerated(
      ReduceImplementation imp, std::string type, std::string input_var,
      std::string output_var, int workgroup_size, std::string larr);

  WorkgroupCodeBlock getScanCode(std::string input_variable,
                                 std::string output_variable,
                                 int workgroup_size, std::string larr = "temp");

  WorkgroupCodeBlock getScanCodeBuiltIn(std::string input_variable,
                                        std::string output_variable);

  WorkgroupCodeBlock getScanCodeGenerated(ReduceImplementation implementation,
                                          std::string input_variable,
                                          std::string output_variable,
                                          int workgroup_size, std::string larr);

  WorkgroupCodeBlock getLowLevelReduction(ReduceImplementation implementation,
                                          std::string operation,
                                          ReduceDirection direction,
                                          int workgroup_size, std::string larr);

  WorkgroupCodeBlock getLowLevelReductionVar1(std::string operation,
                                              ReduceDirection direction,
                                              int workgroup_size);

  WorkgroupCodeBlock getLowLevelReductionVar2(std::string operation,
                                              ReduceDirection direction,
                                              int workgroup_size);

  WorkgroupCodeBlock getLowLevelReductionWarp(std::string operation,
                                              ReduceDirection direction,
                                              int workgroup_size,
                                              std::string larr);
}

#endif  // OCL_WORKGROUP_UTILS_HPP
