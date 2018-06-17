/*
 * author: henning funke
 * date: 26.07.2016
 */

#include <core/variable_manager.hpp>
#include <query_compilation/code_generators/ocl_workgroup_utils.hpp>

#include <iomanip>
#include <list>
#include <set>
#include <sstream>
#include <string>

namespace CoGaDB {

ReduceImplementation getReduceImplementation() {
  std::string workgroup_implementation =
      VariableManager::instance().getVariableValueString(
          "code_gen.ocl.workgroup_function_implementation");
  ReduceImplementation implementation = VAR1;
  if (workgroup_implementation == "var1") {
    implementation = VAR1;
  } else if (workgroup_implementation == "var2") {
    implementation = VAR2;
  } else if (workgroup_implementation == "warp") {
    implementation = WARP;
  }
  return implementation;
}

bool isInclusiveImplementation() {
  if (getReduceImplementation() == WARP) {
    return true;
  } else {
    return false;
  }
}

std::string getGroupResultOffset() {
  if (getReduceImplementation() == WARP) {
    return "32";
  } else {
    return "localSize";
  }
}

std::string getGroupVariableIndex() {
  if (getReduceImplementation() == WARP) {
    return "warpId";
  } else {
    return "0";
  }
}

WorkgroupCodeBlock getScanCode(std::string input_variable,
                               std::string output_variable, int workgroup_size,
                               std::string larr) {
  ReduceImplementation implementation = getReduceImplementation();

  bool builtin = VariableManager::instance().getVariableValueBoolean(
      "code_gen.ocl.use_builtin_workgroup_functions");
  WorkgroupCodeBlock workgroup_scan;
  if (builtin) {
    workgroup_scan = getScanCodeBuiltIn(input_variable, output_variable);
  } else {
    workgroup_scan = getScanCodeGenerated(
        implementation, input_variable, output_variable, workgroup_size, larr);
  }
  return workgroup_scan;
}

WorkgroupCodeBlock getReduceCode(ReduceImplementation imp, std::string type,
                                 std::string input_var, std::string output_var,
                                 int workgroup_size, std::string larr) {
  ReduceImplementation implementation = getReduceImplementation();
  bool builtin = VariableManager::instance().getVariableValueBoolean(
      "code_gen.ocl.use_builtin_workgroup_functions");
  WorkgroupCodeBlock workgroup_reduce;
  if (builtin) {
    return workgroup_reduce;
  } else {
    workgroup_reduce = getReduceCodeGenerated(implementation, type, input_var,
                                              output_var, workgroup_size, larr);
  }
  return workgroup_reduce;
}

WorkgroupCodeBlock getLowLevelReduction(ReduceImplementation implementation,
                                        std::string operation,
                                        ReduceDirection direction,
                                        int workgroup_size, std::string larr) {
  WorkgroupCodeBlock result;
  if (implementation == VAR1) {
    result = getLowLevelReductionVar1(operation, direction, workgroup_size);
  } else if (implementation == VAR2) {
    result = getLowLevelReductionVar2(operation, direction, workgroup_size);
  } else if (implementation == WARP) {
    result =
        getLowLevelReductionWarp(operation, direction, workgroup_size, larr);
  }
  return result;
}

WorkgroupCodeBlock getScanCodeBuiltIn(std::string input_variable,
                                      std::string output_variable) {
  std::stringstream c;
  c << "uint32_t " << output_variable << " = work_group_scan_exclusive_add("
    << input_variable << ");" << std::endl;
  WorkgroupCodeBlock workgroup_scan;
  workgroup_scan.computation = c.str();
  return workgroup_scan;
}

std::string getCodeClear(ReduceImplementation implementation,
                         std::string input_variable, std::string larr) {
  std::stringstream clear_op;
  if (implementation != WARP) {
    clear_op << "if ((lid+1)%" << getGroupResultOffset() << "==0) {"
             << std::endl
             << "  group_total = " << larr << "[locPos];" << std::endl
             << "  " << larr << "[locPos] = 0;" << std::endl
             << "};" << std::endl;
  } else {
    clear_op << "if ((lid+1)%" << getGroupResultOffset() << "==0) {"
             << std::endl
             << "  group_total = " << larr << "[locPos];" << std::endl
             << "};" << std::endl
             << "" << larr << "[locPos] -= " << input_variable << ";"
             << std::endl;
  }
  return clear_op.str();
}

std::string getGlobalInit(ReduceImplementation imp) {
  std::stringstream code;

  if (imp == VAR1) {
    code << "#define WARP_SHIFT 4" << std::endl
         << "#define GRP_SHIFT 8" << std::endl
         << "#define BANK_OFFSET(n)     ((n) >> WARP_SHIFT + (n) >> GRP_SHIFT)"
         << std::endl;
  }
  return code.str();
}

std::string getKernelInit(ReduceImplementation imp) {
  std::stringstream code;

  code << "uint locPos = lid;" << std::endl << "int group_total;" << std::endl;

  if (imp == WARP) {
    code << "int warpId = lid / 32;" << std::endl
         << "uint warpLane = lid & 31;" << std::endl;
  }
  return code.str();
}

WorkgroupCodeBlock getReduceCodeGenerated(
    ReduceImplementation imp, std::string type, std::string input_var,
    std::string output_var, int workgroup_size, std::string larr) {
  std::stringstream local_declaration;
  std::string global_init = getGlobalInit(imp);
  std::string kernel_init = getKernelInit(imp);

  std::stringstream local_init;
  local_init << "__local " << type << " " << larr << "[" << workgroup_size
             << "];" << std::endl;

  std::stringstream reduce_code;
  reduce_code << larr << "[locPos] = " << input_var << ";" << std::endl;
  std::stringstream up_op;
  up_op << "" << larr << "[bi] += " << larr << "[ai];" << std::endl;
  WorkgroupCodeBlock upsweep;
  upsweep =
      getLowLevelReduction(imp, up_op.str(), UPSWEEP, workgroup_size, larr);

  WorkgroupCodeBlock workgroup_reduce;
  workgroup_reduce.global_init = global_init;
  workgroup_reduce.kernel_init = kernel_init;
  workgroup_reduce.local_init = local_init.str() + upsweep.local_init;
  workgroup_reduce.computation = reduce_code.str();
  return workgroup_reduce;
}

WorkgroupCodeBlock getScanCodeGenerated(ReduceImplementation implementation,
                                        std::string input_variable,
                                        std::string output_variable,
                                        int workgroup_size, std::string larr) {
  std::stringstream local_declaration;
  std::string global_init = getGlobalInit(implementation);
  std::string kernel_init = getKernelInit(implementation);

  std::stringstream local_init;
  local_init << "__local uint32_t " << larr << "[" << workgroup_size << "];"
             << std::endl;

  std::stringstream scancode;
  scancode << larr << "[locPos] = " << input_variable << ";" << std::endl;
  std::stringstream up_op;
  up_op << "" << larr << "[bi] += " << larr << "[ai];" << std::endl;
  WorkgroupCodeBlock upsweep;
  upsweep = getLowLevelReduction(implementation, up_op.str(), UPSWEEP,
                                 workgroup_size, larr);

  std::stringstream down_op;
  down_op << "uint32_t t = " << larr << "[ai];" << std::endl
          << larr << "[ai] = " << larr << "[bi];" << std::endl
          << larr << "[bi] += t;" << std::endl;
  WorkgroupCodeBlock downsweep;
  if (implementation != WARP)
    downsweep = getLowLevelReduction(implementation, down_op.str(), DOWNSWEEP,
                                     workgroup_size, larr);

  scancode << upsweep.computation
           << getCodeClear(implementation, input_variable, larr)
           << downsweep.computation;
  scancode << "uint32_t " << output_variable << " = " << larr << "[locPos];"
           << std::endl;

  WorkgroupCodeBlock workgroup_scan;
  workgroup_scan.global_init = global_init;
  workgroup_scan.kernel_init = kernel_init;
  workgroup_scan.local_init = local_init.str() + upsweep.local_init;
  workgroup_scan.computation = scancode.str();
  return workgroup_scan;
}

WorkgroupCodeBlock getLowLevelReductionVar1(std::string operation,
                                            ReduceDirection direction,
                                            int workgroup_size) {
  std::stringstream local_init;
  local_init << "int n = localSize;" << std::endl
             << "int ai = lid;" << std::endl
             << "int bi = lid >> 1;" << std::endl
             << "int bankOffsetA = BANK_OFFSET(ai);" << std::endl
             << "int bankOffsetB = BANK_OFFSET(bi);" << std::endl;

  std::stringstream code;
  if (direction == UPSWEEP) {
    code << "int offset=1;" << std::endl
         << "for (int d = n>>1; d > 0; d >>= 1) {" << std::endl
         << "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
         << "    if (lid < d)" << std::endl
         << "    {  " << std::endl
         << "        int ai = offset * (2*lid + 1)-1;" << std::endl
         << "        int bi = offset * (2*lid + 2)-1;" << std::endl
         << "        ai += BANK_OFFSET(ai);" << std::endl
         << "        bi += BANK_OFFSET(bi);" << std::endl;
    code << operation;
    code << "    }  " << std::endl
         << "    offset <<= 1; " << std::endl
         << "}" << std::endl
         << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  } else if (direction == DOWNSWEEP) {
    code << "for (int d = 1; d < n; d <<= 1) {" << std::endl
         << "    offset >>= 1;" << std::endl
         << "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
         << "    if (lid < d)" << std::endl
         << "    {  " << std::endl
         << "        int ai = offset * (2*lid + 1)-1;" << std::endl
         << "        int bi = offset * (2*lid + 2)-1;" << std::endl
         << "        ai += BANK_OFFSET(ai);" << std::endl
         << "        bi += BANK_OFFSET(bi);" << std::endl;
    code << operation;
    code << "    }  " << std::endl
         << "}" << std::endl
         << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  }
  WorkgroupCodeBlock result;
  result.local_init = local_init.str();
  result.computation = code.str();
  return result;
}

WorkgroupCodeBlock getLowLevelReductionVar2(std::string operation,
                                            ReduceDirection direction,
                                            int workgroup_size) {
  std::stringstream kernel_init;
  kernel_init << "int lid = get_local_id(0);" << std::endl
              << "int localSize = get_local_size(0);" << std::endl
              << "int globalIdx = get_group_id(0);" << std::endl;

  std::stringstream local_init;
  local_init << "int n = localSize;" << std::endl
             << "int bi = (lid*2)+1;" << std::endl
             << "int depth = (int) log2((float)localSize);" << std::endl;

  std::stringstream scan_code;
  if (direction == UPSWEEP) {
    scan_code << "for(int d=0; d<depth; d++) {" << std::endl
              << "  barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
              << "  int mask = (0x1 << d) - 1;" << std::endl
              << "  if((lid & mask) == mask && bi < localSize) {" << std::endl
              << "    int offset = (0x1 << d);" << std::endl
              << "    int ai = bi - offset;" << std::endl;
    scan_code << operation;
    scan_code << "  }" << std::endl << "}" << std::endl;
  } else if (direction == DOWNSWEEP) {
    scan_code << "for(int d=depth; d>-1; d--) {" << std::endl
              << "  barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
              << "  int mask = (0x1 << d) - 1;" << std::endl
              << "  if((lid & mask) == mask && bi < localSize) {" << std::endl
              << "    int offset = (0x1 << d);" << std::endl
              << "    int ai = bi - offset;" << std::endl;
    scan_code << operation;
    scan_code << "  }" << std::endl
              <<           // if clause lane selection
        "}" << std::endl;  // for loop downsweep
  }
  scan_code << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

  WorkgroupCodeBlock result;
  result.global_init = "";
  result.kernel_init = kernel_init.str();
  result.local_init = local_init.str();
  result.computation = scan_code.str();
  return result;
}

std::string getCustomScanCode(std::string upsweep_op, std::string clear_op,
                              std::string downsweep_op) {
  // upsweep
  std::stringstream scan_code;
  scan_code << "int n = localSize;" << std::endl
            << "int bi = (lid*2)+1;" << std::endl
            << "int depth = (int) log2((float)localSize);" << std::endl
            << "for(int d=0; d<depth; d++) {" << std::endl
            << "  barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
            << "  int mask = (0x1 << d) - 1;" << std::endl
            << "  if((lid & mask) == mask && bi < localSize) {" << std::endl
            << "    int offset = (0x1 << d);" << std::endl
            << "    int ai = bi - offset;" << std::endl;
  scan_code << upsweep_op;
  scan_code << "  }" << std::endl << "}" << std::endl;

  // clear
  scan_code << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  scan_code << clear_op;

  // downsweep
  scan_code << "for(int d=depth; d>-1; d--) {" << std::endl
            << "  barrier(CLK_LOCAL_MEM_FENCE);" << std::endl
            << "  int mask = (0x1 << d) - 1;" << std::endl
            << "  if((lid & mask) == mask && bi < localSize) {" << std::endl
            << "    int offset = (0x1 << d);" << std::endl
            << "    int ai = bi - offset;" << std::endl;
  scan_code << downsweep_op;
  scan_code << "  }" << std::endl
            <<           // if clause lane selection
      "}" << std::endl;  // for loop downsweep
  scan_code << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  return scan_code.str();
}

WorkgroupCodeBlock getLowLevelReductionWarp(std::string operation,
                                            ReduceDirection direction,
                                            int workgroup_size,
                                            std::string larr) {
  std::stringstream kernel_init;
  std::string local_name = larr;
  std::stringstream scan_code;
  for (int i = 1; i <= 16; i *= 2) {
    scan_code << "if(warpLane >= " << i << ") {" << local_name
              << "[locPos] += " << local_name << "[locPos - " << i << "];"
              << "}" << std::endl;
  }

  WorkgroupCodeBlock result;
  result.global_init = "";
  result.kernel_init = kernel_init.str();
  result.computation = scan_code.str();
  return result;
}

std::string getSegmentedScan(std::map<std::string, std::string> aggVars,
                             std::string data_access_suffix) {
  std::string code;

  if (getReduceImplementation() == WARP) {
    code = getSegmentedReductionWarpInclusive("flags_loc", aggVars,
                                              data_access_suffix);

  } else {
    code = getSegmentedScanWorkgroup(aggVars, data_access_suffix);
  }

  return code;
}

std::string getSegmentedScanWorkgroup(
    std::map<std::string, std::string> aggVars, std::string suffix) {
  // segmented reduction
  std::stringstream seg_upsweep, seg_clear, seg_downsweep;
  // upsweep
  seg_upsweep << "if(!flags_loc[bi]) {" << std::endl;
  for (auto& var : aggVars) {
    seg_upsweep << var.first << suffix << "[bi] += " << var.first << ""
                << suffix << "[ai];" << std::endl;
  }
  seg_upsweep << "}" << std::endl;
  seg_upsweep << "flags_loc[bi] = flags_loc[bi] | flags_loc[ai];" << std::endl;
  // clear
  seg_clear << "if(lid == localSize-1) {" << std::endl;
  for (auto& var : aggVars) {
    seg_clear << var.first << "" << suffix << "[localSize-1] = (" << var.second
              << ") 0;" << std::endl;
  }
  seg_clear << "}" << std::endl;
  // downsweep
  for (auto& var : aggVars) {
    seg_downsweep << var.second << " " << var.first << "_tmp = " << var.first
                  << "" << suffix << "[ai];" << std::endl;
    seg_downsweep << var.first << "" << suffix << "[ai] = " << var.first << ""
                  << suffix << "[bi];" << std::endl;
  }
  seg_downsweep << "if(start_flags_loc[ai+1]) {" << std::endl;
  for (auto& var : aggVars) {
    seg_downsweep << var.first << "" << suffix << "[bi] = (" << var.second
                  << ") 0;" << std::endl;
  }
  seg_downsweep << "} else if(flags_loc[ai]) {" << std::endl;
  for (auto& var : aggVars) {
    seg_downsweep << var.first << "" << suffix << "[bi] = " << var.first
                  << "_tmp;" << std::endl;
  }
  seg_downsweep << "} else {" << std::endl;
  for (auto& var : aggVars) {
    seg_downsweep << var.first << "" << suffix << "[bi] += " << var.first
                  << "_tmp;" << std::endl;
  }
  seg_downsweep << "}" << std::endl;
  seg_downsweep << "flags_loc[ai] = 0;" << std::endl;
  std::string reduce_segments_code = getCustomScanCode(
      seg_upsweep.str(), seg_clear.str(), seg_downsweep.str());
  return reduce_segments_code;
}

std::string getSegmentedReductionWarpInclusive(
    std::string head_var, std::map<std::string, std::string> aggVars,
    std::string suffix) {
  std::stringstream kernel_init;
  std::stringstream local_init;
  std::string head = head_var + "[locPos]";
  std::string op = " + ";
  std::stringstream scan_code;
  for (int i = 1; i <= 16; i *= 2) {
    scan_code << "if(warpLane >= " << i << ") {" << std::endl;

    for (auto var : aggVars) {
      std::string data_var = var.first + suffix;
      std::string data = data_var + "[locPos]";
      // conditional scan
      scan_code << data << " = " << head << " ? " << data << " : " << data_var
                << "[locPos - " << i << "]" << op << data << ";" << std::endl;
    }
    // head flag update
    scan_code << head << " = " << head_var << "[locPos - " << i << "] | "
              << head << ";" << std::endl;
    scan_code << "}" << std::endl;
  }

  WorkgroupCodeBlock result;
  result.global_init = "";
  result.kernel_init = kernel_init.str();
  result.local_init = local_init.str();
  result.computation = scan_code.str();
  return scan_code.str();  // result;
}

}  // namespace CoGaDB
