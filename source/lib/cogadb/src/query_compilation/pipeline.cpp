
#include <query_compilation/pipeline.hpp>

#include <query_compilation/minimal_api_c.h>
#include <query_compilation/minimal_api_c_internal.hpp>

#include <dlfcn.h>
#include <boost/filesystem.hpp>

#include <util/time_measurement.hpp>

#include <boost/bind.hpp>
#include <boost/uuid/uuid.hpp>             // uuid class
#include <boost/uuid/uuid_generators.hpp>  // generators
#include <boost/uuid/uuid_io.hpp>          // streaming operators etc.
#include <boost/version.hpp>

#include <boost/bind/placeholders.hpp>
// workaround change of boost::placeholder from version 1.60
#if BOOST_VERSION >= 106000
using namespace boost::placeholders;
#endif

#include <boost/make_shared.hpp>

#include <sstream>

#include <core/base_table.hpp>
#include <core/variable_manager.hpp>
#include <query_compilation/pipeline_info.hpp>
#include <util/shared_library.hpp>

#include <query_compilation/code_generators/c_code_compiler.hpp>
#include "query_compilation/pipeline_job.hpp"

namespace CoGaDB {

Pipeline::Pipeline(double compile_time_in_sec, const ScanParam& scan_param,
                   PipelineInfoPtr pipe_info)
    : mPipeInfo(pipe_info),
      mScanParam(scan_param),
      mResult(),
      mCompileTimeInSeconds(compile_time_in_sec),
      mBegin(0),
      mEnd(0) {}

Pipeline::Pipeline(const Pipeline& other)
    : mPipeInfo(boost::make_shared<PipelineInfo>(*other.mPipeInfo)),
      mScanParam(other.mScanParam),
      mResult(other.mResult),
      mCompileTimeInSeconds(other.mCompileTimeInSeconds),
      mBegin(other.mBegin),
      mEnd(other.mEnd) {}

Pipeline::~Pipeline() {}

bool Pipeline::execute() {
  mBegin = getTimestamp();
  mResult = execute_impl();
  mEnd = getTimestamp();

  if (mResult) {
    return true;
  } else {
    return false;
  }
}

TablePtr SharedLibPipeline::execute_impl() {
  if (mQuery) {
    return mQuery(mScanParam, mPipeInfo->getGlobalState());
  } else {
    return TablePtr();
  }
}

TablePtr Pipeline::getResult() const { return mResult; }

const ScanParam Pipeline::getScannedAttributes() const { return mScanParam; }

const PipelineInfoPtr Pipeline::getPipelineInfo() const { return mPipeInfo; }

double Pipeline::getCompileTimeSec() const {
  return this->getHostCompileTimeSec() + this->getKernelCompileTimeSec();
}

double Pipeline::getHostCompileTimeSec() const { return mCompileTimeInSeconds; }
double Pipeline::getKernelCompileTimeSec() const { return 0.0; }

double Pipeline::getExecutionTimeSec() const {
  if (!mResult) {
    return double(0);
  }

  assert(mEnd >= mBegin);
  double exec_time = double(mEnd - mBegin) / (1000 * 1000 * 1000);

  return exec_time;
}

bool Pipeline::replaceInputTables(
    const std::map<TablePtr, TablePtr>& table_replacement) {
  ScanParam copy;
  std::map<TablePtr, TablePtr>::const_iterator cit;
  for (size_t i = 0; i < mScanParam.size(); ++i) {
    cit = table_replacement.find(mScanParam[i].getTable());
    if (cit == table_replacement.end()) {
      copy.push_back(mScanParam[i]);
    } else {
      assert(cit->first != NULL);
      assert(cit->second != NULL);
      AttributeReferencePtr new_attr =
          createInputAttributeForNewTable(mScanParam[i], cit->second);
      assert(new_attr != NULL);
      copy.push_back(*new_attr);
    }
  }

  mScanParam.clear();
  mScanParam.insert(mScanParam.begin(), copy.begin(), copy.end());
  return true;
}

const std::string Pipeline::toString() const {
  std::stringstream str;
  str << "Compile Time: " << mCompileTimeInSeconds << "s" << std::endl;
  str << "Execution Time: " << mCompileTimeInSeconds << "s" << std::endl;
  str << "Result: " << (void*)mResult.get() << std::endl;
  str << "Scan Param: " << std::endl;
  for (size_t i = 0; i < mScanParam.size(); ++i) {
    str << "\t" << CoGaDB::toString(mScanParam[i]) << ": "
        << (void*)mScanParam[i].getTable().get() << std::endl;
  }
  return str.str();
}

SharedLibPipeline::SharedLibPipeline(
    PipelineQueryFunction query, double compile_time_in_sec,
    SharedLibraryPtr shared_lib, const ScanParam& scan_param,
    PipelineInfoPtr pipe_info, const std::string& compiled_query_base_file_name)
    : FunctionPipeline(query, compile_time_in_sec, scan_param, pipe_info),
      mSharedLibrary(shared_lib),
      mCompiledQueryBaseFileName(compiled_query_base_file_name) {
  assert(mSharedLibrary != NULL);
}

SharedLibPipeline::SharedLibPipeline(const SharedLibPipeline& other)
    : FunctionPipeline(other),
      mSharedLibrary(other.mSharedLibrary),
      mCompiledQueryBaseFileName(other.mCompiledQueryBaseFileName) {}

SharedLibPipeline::~SharedLibPipeline() {
  if (VariableManager::instance().getVariableValueBoolean(
          "cleanup_generated_files")) {
    if (boost::filesystem::exists(mCompiledQueryBaseFileName + ".cpp")) {
      boost::filesystem::remove(mCompiledQueryBaseFileName + ".cpp");
    }

    if (boost::filesystem::exists(mCompiledQueryBaseFileName + ".c")) {
      boost::filesystem::remove(mCompiledQueryBaseFileName + ".c");
    }

    if (boost::filesystem::exists(mCompiledQueryBaseFileName + ".o")) {
      boost::filesystem::remove(mCompiledQueryBaseFileName + ".o");
    }

    if (boost::filesystem::exists(mCompiledQueryBaseFileName + ".so")) {
      boost::filesystem::remove(mCompiledQueryBaseFileName + ".so");
    }

    if (boost::filesystem::exists(mCompiledQueryBaseFileName + ".cpp.orig")) {
      boost::filesystem::remove(mCompiledQueryBaseFileName + ".cpp.orig");
    }

    if (boost::filesystem::exists(mCompiledQueryBaseFileName + ".c.orig")) {
      boost::filesystem::remove(mCompiledQueryBaseFileName + ".c.orig");
    }
  }
}

SharedCppLibPipeline::SharedCppLibPipeline(
    SharedCppLibPipelineQueryPtr query, const ScanParam& scan_param,
    double compile_time_in_sec, SharedLibraryPtr shared_lib,
    PipelineInfoPtr pipe_info, const std::string& compiled_query_base_file_name)
    : SharedLibPipeline(query, compile_time_in_sec, shared_lib, scan_param,
                        pipe_info, compiled_query_base_file_name) {}

SharedCppLibPipeline::SharedCppLibPipeline(const SharedCppLibPipeline& other)
    : SharedLibPipeline(other) {}

const PipelinePtr SharedCppLibPipeline::copy() const {
  return PipelinePtr(new SharedCppLibPipeline(*this));
}

SharedCLibPipeline::SharedCLibPipeline(
    SharedCLibPipelineQueryPtr query, const ScanParam& scan_param,
    double compile_time_in_sec, SharedLibraryPtr shared_lib,
    PipelineInfoPtr pipe_info, const std::string& compiled_query_base_file_name)
    : SharedLibPipeline(boost::bind(query_wrapper, query, _1, _2),
                        compile_time_in_sec, shared_lib, scan_param, pipe_info,
                        compiled_query_base_file_name) {}

SharedCLibPipeline::SharedCLibPipeline(const SharedCLibPipeline& other)
    : SharedLibPipeline(other) {}

const PipelinePtr SharedCLibPipeline::copy() const {
  return PipelinePtr(new SharedCLibPipeline(*this));
}

const TablePtr SharedCLibPipeline::query_wrapper(
    SharedCLibPipelineQueryPtr query, const ScanParam& param, StatePtr state) {
  (void)state;
  C_Table** c_tables = (C_Table**)malloc(param.size() * sizeof(C_Table*));

  for (unsigned int i = 0; i < param.size(); ++i) {
    c_tables[i] = getCTableFromTablePtr(param[i].getTable());
  }

  C_Table* c_table = (*query)(c_tables);

  if (c_table == NULL) {
    return TablePtr();
  }

  TablePtr table = getTablePtrFromCTable(c_table);
  releaseTable(c_table);

  for (unsigned int i = 0; i < param.size(); ++i) {
    releaseTable(c_tables[i]);
  }
  free(c_tables);

  return table;
}

LLVMJitPipeline::LLVMJitPipeline(
    SharedCLibPipelineQueryPtr query, const ScanParam& scan_param,
    double compile_time_in_sec,
    const boost::shared_ptr<llvm::LLVMContext>& context,
    const boost::shared_ptr<llvm::ExecutionEngine>& engine,
    PipelineInfoPtr pipe_info)
    : FunctionPipeline(
          boost::bind(SharedCLibPipeline::query_wrapper, query, _1, _2),
          compile_time_in_sec, scan_param, pipe_info),
      mContext(context),
      mEngine(engine) {}

LLVMJitPipeline::LLVMJitPipeline(const LLVMJitPipeline& other)
    : FunctionPipeline(other) {}

TablePtr LLVMJitPipeline::execute_impl() {
  if (mQuery) {
    return mQuery(mScanParam, mPipeInfo->getGlobalState());
  } else {
    return TablePtr();
  }
}

const PipelinePtr LLVMJitPipeline::copy() const {
  return PipelinePtr(new LLVMJitPipeline(*this));
}

CPipeline::CPipeline(const ScanParam& scan_param,
                     const PipelineInfoPtr& pipe_info,
                     CompiledCCodePtr compiled_code)
    : Pipeline(compiled_code->getCompileTimeInSeconds(), scan_param, pipe_info),
      compiled_code_(compiled_code) {}

CPipeline::CPipeline(const CPipeline& other)
    : Pipeline(other), compiled_code_(other.compiled_code_) {}

const PipelinePtr CPipeline::copy() const {
  return PipelinePtr(new CPipeline(*this));
}

TablePtr CPipeline::execute_impl() {
  std::vector<C_Table*> c_tables(mScanParam.size());

  for (unsigned int i = 0; i < mScanParam.size(); ++i) {
    c_tables[i] = getCTableFromTablePtr(mScanParam[i].getTable());
  }

  auto c_table = callCFunction(c_tables.data());

  if (c_table == NULL) {
    return nullptr;
  }

  TablePtr table = getTablePtrFromCTable(c_table);
  releaseTable(c_table);

  for (auto& c_table : c_tables) {
    releaseTable(c_table);
  }

  return table;
}

C_Table* CPipeline::callCFunction(C_Table** c_tables) {
  return (*compiled_code_->getFunctionPointer<SharedCLibPipelineQueryPtr>(
      "compiled_query"))(c_tables);
}

OCLPipeline::OCLPipeline(const ScanParam& scan_param,
                         const PipelineInfoPtr& pipe_info,
                         CompiledCCodePtr compiled_code,
                         double opencl_compile_time,
                         OCL_Execution_ContextPtr execution_context)
    : CPipeline(scan_param, pipe_info, compiled_code),
      opencl_compile_time_(opencl_compile_time),
      execution_context_(execution_context) {}

OCLPipeline::OCLPipeline(const OCLPipeline& other)
    : CPipeline(other),
      opencl_compile_time_(other.opencl_compile_time_),
      execution_context_(other.execution_context_) {}

const PipelinePtr OCLPipeline::copy() const {
  return PipelinePtr(new OCLPipeline(*this));
}

// double OCLPipeline::getCompileTimeSec() const {
//  return Pipeline::getCompileTimeSec() + opencl_compile_time_;
//}

double OCLPipeline::getKernelCompileTimeSec() const {
  return this->opencl_compile_time_;
}

C_Table* OCLPipeline::callCFunction(C_Table** c_tables) {
  return (*compiled_code_->getFunctionPointer<OCLPipelineQueryPtr>(
      "compiled_query"))(c_tables, execution_context_.get(), NULL);
}

DummyPipeline::DummyPipeline(TablePtr result, const ScanParam& scan_param,
                             PipelineInfoPtr pipe_info)
    : Pipeline(0, scan_param, pipe_info) {
  mResult = result;
}

DummyPipeline::DummyPipeline(const DummyPipeline& other) : Pipeline(other) {}

TablePtr DummyPipeline::execute_impl() { return mResult; }

const PipelinePtr DummyPipeline::copy() const {
  return PipelinePtr(new DummyPipeline(*this));
}

const PipelinePtr compileQueryFile(const std::string& path,
                                   const ScanParam& param) {
  int ret = 0;

  if (!boost::filesystem::exists(path)) {
    std::cerr << "Could not find file: '" << path << "'" << std::endl;
    return PipelinePtr();
  }

  ScanParam scan_param;
  scan_param.insert(scan_param.begin(), param.begin(), param.end());

  boost::uuids::uuid uuid = boost::uuids::random_generator()();
  std::stringstream ss;
  ss << "gen_query_" << uuid;

  std::string filename = ss.str() + ".cpp";

  std::string copy_query_file_command =
      std::string("cp '") + path + std::string("' ") + filename;
  ret = system(copy_query_file_command.c_str());

  std::string format_command = std::string("astyle ") + filename;
  ret = system(format_command.c_str());

  std::string copy_last_query_command =
      std::string("cp '") + filename +
      std::string("' last_generated_query.cpp");
  ret = system(copy_last_query_command.c_str());

  Timestamp begin_compile = getTimestamp();

  std::stringstream compile_command;
  compile_command << "clang -g -O3 -I ../cogadb/include/ -I "
                     "../hype-library/include/ -c -fpic ";
  compile_command << filename << " -o " << ss.str() << ".o";
  ret = system(compile_command.str().c_str());
  if (ret != 0) {
    std::cout << "Compilation Failed!" << std::endl;
    return PipelinePtr();
  } else {
    std::cout << "Compilation Successful!" << std::endl;
  }
  std::stringstream linking_command;
  linking_command << "g++ -shared " << ss.str() << ".o -o " << ss.str() << ".so"
                  << std::endl;
  ret = system(linking_command.str().c_str());

  Timestamp end_compile = getTimestamp();

  std::stringstream shared_lib_path;
  shared_lib_path << "./" << ss.str() << ".so";

  SharedLibraryPtr shared_lib = SharedLibrary::load(shared_lib_path.str());
  assert(shared_lib != NULL);

  std::string symbol_name =
      "_Z14compiled_queryRKSt6vectorIN6CoGaDB18AttributeReferenceESaIS1_"
      "EEN5boost10shared_ptrINS0_5StateEEE";
  SharedCppLibPipelineQueryPtr query =
      shared_lib->getFunction<SharedCppLibPipelineQueryPtr>(symbol_name);
  assert(query != NULL);

  /* in case no scan parameter was passed, fetch a pointer to a generator
   * function that returns the scan parameter for this query */
  if (scan_param.empty()) {
    /* get pointer to function "const ScanParam getScanParam()" */
    ScanParamGeneratorPtr scan_param_generator =
        shared_lib->getFunction<ScanParamGeneratorPtr>("_Z12getScanParamv");
    assert(scan_param_generator != NULL);
    scan_param = (*scan_param_generator)();
  }
  assert(!scan_param.empty());
  PipelineInfoPtr pipe_info = boost::make_shared<PipelineInfo>();
  pipe_info->setSourceTable(scan_param.front().getTable());
  //        pipe_info->setPipelineType(this->pipe_end);
  //        pipe_info->setGroupByAggregateParam(this->groupby_param);

  double compile_time_in_sec =
      double(end_compile - begin_compile) / (1000 * 1000 * 1000);
  return PipelinePtr(new SharedCppLibPipeline(
      query, scan_param, compile_time_in_sec, shared_lib, pipe_info, ss.str()));
}

const TablePtr compileAndExecuteQueryFile(
    const std::string& path_to_query_file) {
  PipelinePtr pipe = compileQueryFile("test_query_hand_coded.cpp");
  if (!pipe) {
    return TablePtr();
  }
  if (!pipe->execute()) {
    return TablePtr();
  }
  return pipe->getResult();
}

bool isDummyPipeline(PipelinePtr pipeline) {
  if (!pipeline) return false;
  if (boost::dynamic_pointer_cast<DummyPipeline>(pipeline)) {
    return true;
  } else {
    return false;
  }
}

}  // end namespace CoGaDB
