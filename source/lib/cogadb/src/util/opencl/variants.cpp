
#include <util/variant_configurator.hpp>

namespace CoGaDB {

// struct VariantMeasurement{
// VariantMeasurement(bool _success, double exec_time, double kernel_comp_time,
// double host_comp_time);
// bool success;
// double total_execution_time_in_s;
// double total_kernel_compilation_time_in_s;
// double total_host_compilation_time_in_s;
//};

// VariantMeasurement::VariantMeasurement(bool _success,
// double _exec_time,
// double _kernel_comp_time,
// double _host_comp_time)
//: success(_success), total_execution_time_in_s(_exec_time),
// total_kernel_compilation_time_in_s(_kernel_comp_time),
// total_host_compilation_time_in_s(_host_comp_time)
//{}

// const VariantMeasurement executeQueryWithVariants(const std::string& query){
//  VariantMeasurement vm(true, 0, 0, 0);
//  VariantIterator vit;
//  VariantConfigurator vc;
//  vit.add("code_gen.pipe_exec_strategy",
//          {"parallel_global_atomic_single_pass",
//           "serial_single_pass",
//           "parallel_three_pass"});
//  vit.add("code_gen.memory_access",
//          {"coalesced", "sequential"});
//  vit.add("code_gen.opt.enable_predication",
//          {"false", "true"});

//  while(vit.hasNext()){
//    vc(vit.getNext());

//    testResultCorrect(deviceDir, GetParam().second);
//  } );

//  return vm;
//}

}  // end namespace CoGaDB
