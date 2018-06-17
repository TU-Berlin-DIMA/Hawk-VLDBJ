#include <query_compilation/kernel_execution.hpp>
#include <sstream>

KernelExecution::KernelExecution() {}

void KernelExecution::init(std::string name, int block_size,
                           std::string index_variable_type,
                           std::string tuple_id_var_name, size_t limit,
                           bool use_selected) {
  this->name = name;
  this->block_size = block_size;
  this->index_variable_type = index_variable_type;
  this->tuple_id_var_name = tuple_id_var_name;
  this->limit = limit;
  this->use_selected = use_selected;
  this->codeIsProduced = false;

  if (limit > 0) {
    std::stringstream limit_clause;
    limit_clause << "if(thread_index < " << limit << ") {" << std::endl;
    if (use_selected) limit_clause << "selected = true;";
    kernel_code_upper.push_back(limit_clause.str());
  }
}

void KernelExecution::addParameter(std::string type, std::string param_name) {
  parameter_types.push_back(type);
  parameter_names.push_back(param_name);
  parameter_call_expressions.push_back("d_" + param_name);
}

void KernelExecution::addParameter(std::string type, std::string param_name,
                                   std::string call_expression) {
  parameter_types.push_back(type);
  parameter_names.push_back(param_name);
  parameter_call_expressions.push_back(call_expression);
}

void KernelExecution::produceFrameCode() {
  if (codeIsProduced) return;

  std::stringstream functionHeader;
  functionHeader << "__global__ void " << name << "(";

  functionHeader << parameter_types[0] << " " << parameter_names[0];

  for (size_t i = 1; i < parameter_types.size(); i++) {
    functionHeader << ", " << parameter_types[i] << " " << parameter_names[i];
  }

  functionHeader << ") {" << std::endl;

  functionHeader << index_variable_type
                 << " block_index = blockIdx.y * gridDim.x + blockIdx.x;"
                 << std::endl;
  functionHeader << index_variable_type
                 << " thread_index = threadIdx.x + blockIdx.x * blockDim.x"
                 << " + blockIdx.y * blockDim.x * gridDim.x;" << std::endl;

  functionHeader << index_variable_type << " " << tuple_id_var_name
                 << " = thread_index;" << std::endl;

  if (use_selected) functionHeader << "bool selected = false;" << std::endl;

  function_header = functionHeader.str();

  codeIsProduced = true;
}

void KernelExecution::prepend_upper(std::string line) {
  kernel_code_upper.push_front(line);
}

void KernelExecution::append_upper(std::string line) {
  kernel_code_upper.push_back(line);
}

void KernelExecution::append_lower(std::string line) {
  kernel_code_lower.push_back(line);
}

int KernelExecution::getBlockSize() { return block_size; }

std::string KernelExecution::getInvocationCode() {
  std::stringstream kernel_call;
  kernel_call << name << "<<<grid, " << block_size << ">>>(";

  kernel_call << "d_" << parameter_names[0];
  for (size_t i = 1; i < parameter_names.size(); i++) {
    kernel_call << ", " << parameter_call_expressions[i];
  }

  kernel_call << ");" << std::endl;
  kernel_call << "gpuErrchk(cudaDeviceSynchronize());" << std::endl;

  // gpuErrchk(ans)

  return kernel_call.str();
}

std::string KernelExecution::getKernelCode() {
  produceFrameCode();

  std::stringstream out_kernel;

  out_kernel << function_header;

  std::list<std::string>::const_iterator cit;
  for (cit = this->kernel_code_upper.begin();
       cit != this->kernel_code_upper.end(); ++cit) {
    out_kernel << *cit << std::endl;
  }

  // close selection clause
  if (limit > 0) out_kernel << "}" << std::endl;

  for (cit = this->kernel_code_lower.begin();
       cit != this->kernel_code_lower.end(); ++cit) {
    out_kernel << *cit << std::endl;
  }

  // close kernel
  out_kernel << "}" << std::endl;

  return out_kernel.str();
}
