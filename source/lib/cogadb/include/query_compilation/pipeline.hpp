/*
 * File:   pipeline.hpp
 * Author: sebastian
 *
 * Created on 21. August 2015, 14:56
 */

#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <boost/function.hpp>
#include <core/attribute_reference.hpp>
#include <util/time_measurement.hpp>

struct C_Table;
struct C_State;
struct OCL_Execution_Context;
typedef boost::shared_ptr<OCL_Execution_Context> OCL_Execution_ContextPtr;

namespace llvm {
  class ExecutionEngine;
  class LLVMContext;
}

namespace CoGaDB {

  class State;
  typedef boost::shared_ptr<State> StatePtr;
  typedef const ScanParam (*ScanParamGeneratorPtr)();
  typedef boost::function<const TablePtr(const ScanParam&, StatePtr)>
      PipelineQueryFunction;

  class Pipeline;
  typedef boost::shared_ptr<Pipeline> PipelinePtr;

  class PipelineInfo;
  typedef boost::shared_ptr<PipelineInfo> PipelineInfoPtr;

  class QueryContext;
  typedef boost::shared_ptr<QueryContext> QueryContextPtr;

  class Pipeline {
   public:
    Pipeline(double compile_time_in_sec, const ScanParam& scan_param,
             PipelineInfoPtr pipe_info);
    bool execute();

    TablePtr getResult() const;
    virtual double getCompileTimeSec() const;
    double getHostCompileTimeSec() const;
    virtual double getKernelCompileTimeSec() const;
    double getExecutionTimeSec() const;
    const PipelineInfoPtr getPipelineInfo() const;
    const ScanParam getScannedAttributes() const;
    virtual ~Pipeline();
    virtual bool replaceInputTables(
        const std::map<TablePtr, TablePtr>& table_replacement);
    virtual const PipelinePtr copy() const = 0;
    virtual const std::string toString() const;

   protected:
    virtual TablePtr execute_impl() = 0;
    Pipeline(const Pipeline&);
    Pipeline& operator=(const Pipeline&) = delete;

    PipelineInfoPtr mPipeInfo;
    ScanParam mScanParam;
    TablePtr mResult;

   private:
    double mCompileTimeInSeconds;
    Timestamp mBegin;
    Timestamp mEnd;
  };

  class FunctionPipeline : public Pipeline {
   public:
    FunctionPipeline(PipelineQueryFunction query, double compile_time_in_sec,
                     const ScanParam& scan_param, PipelineInfoPtr pipe_info)
        : Pipeline(compile_time_in_sec, scan_param, pipe_info), mQuery(query) {}

    FunctionPipeline(const FunctionPipeline& other)
        : Pipeline(other), mQuery(other.mQuery) {}

   protected:
    PipelineQueryFunction mQuery;

   private:
  };

  class SharedLibrary;
  typedef boost::shared_ptr<SharedLibrary> SharedLibraryPtr;

  class SharedLibPipeline : public FunctionPipeline {
   public:
    SharedLibPipeline(PipelineQueryFunction query, double compile_time_in_sec,
                      SharedLibraryPtr shared_lib, const ScanParam& scan_param,
                      PipelineInfoPtr pipe_info,
                      const std::string& compiled_query_base_file_name);

    virtual ~SharedLibPipeline();

   protected:
    virtual TablePtr execute_impl();
    SharedLibPipeline(const SharedLibPipeline&);

   private:
    SharedLibraryPtr mSharedLibrary;
    std::string mCompiledQueryBaseFileName;
  };

  typedef TablePtr (*SharedCppLibPipelineQueryPtr)(const ScanParam&, StatePtr);

  class SharedCppLibPipeline : public SharedLibPipeline {
   public:
    SharedCppLibPipeline(SharedCppLibPipelineQueryPtr query,
                         const ScanParam& scan_param,
                         double compile_time_in_sec,
                         SharedLibraryPtr shared_lib, PipelineInfoPtr pipe_info,
                         const std::string& compiled_query_base_file_name);

    virtual const PipelinePtr copy() const;

   private:
    SharedCppLibPipeline(const SharedCppLibPipeline& other);
  };

  typedef C_Table* (*SharedCLibPipelineQueryPtr)(C_Table**);

  class SharedCLibPipeline : public SharedLibPipeline {
   public:
    SharedCLibPipeline(SharedCLibPipelineQueryPtr query,
                       const ScanParam& scan_param, double compile_time_in_sec,
                       SharedLibraryPtr shared_lib, PipelineInfoPtr pipe_info,
                       const std::string& compiled_query_base_file_name);

    virtual const PipelinePtr copy() const;

    static const TablePtr query_wrapper(SharedCLibPipelineQueryPtr query,
                                        const ScanParam& param, StatePtr state);

   private:
    SharedCLibPipeline(const SharedCLibPipeline& other);
  };

  class LLVMJitPipeline : public FunctionPipeline {
   public:
    LLVMJitPipeline(SharedCLibPipelineQueryPtr query,
                    const ScanParam& scan_param, double compile_time_in_sec,
                    const boost::shared_ptr<llvm::LLVMContext>& context,
                    const boost::shared_ptr<llvm::ExecutionEngine>& engine,
                    PipelineInfoPtr pipe_info);

    virtual const PipelinePtr copy() const;

   private:
    virtual TablePtr execute_impl();
    LLVMJitPipeline(const LLVMJitPipeline& other);

    boost::shared_ptr<llvm::LLVMContext> mContext;
    boost::shared_ptr<llvm::ExecutionEngine> mEngine;
  };

  class CompiledCCode;
  typedef boost::shared_ptr<CompiledCCode> CompiledCCodePtr;

  class CPipeline : public Pipeline {
   public:
    CPipeline(const ScanParam& scan_param, const PipelineInfoPtr& pipe_info,
              CompiledCCodePtr compiled_code);

    const PipelinePtr copy() const override;

   protected:
    CPipeline(const CPipeline& other);

    TablePtr execute_impl() override final;
    virtual C_Table* callCFunction(C_Table** c_tables);

    CompiledCCodePtr compiled_code_;
  };

  typedef C_Table* (*OCLPipelineQueryPtr)(C_Table**, OCL_Execution_Context*,
                                          C_State*);

  class OCLPipeline : public CPipeline {
   public:
    OCLPipeline(const ScanParam& scan_param, const PipelineInfoPtr& pipe_info,
                CompiledCCodePtr compiled_code, double opencl_compile_time,
                OCL_Execution_ContextPtr execution_context);

    const PipelinePtr copy() const override;

    double getKernelCompileTimeSec() const override;

   private:
    OCLPipeline(const OCLPipeline& other);

    C_Table* callCFunction(C_Table** c_tables) override;

    double opencl_compile_time_;
    OCL_Execution_ContextPtr execution_context_;
  };
  /* \brief the DummyPipeline is used to omit compilation steps for empty
   * pipelines and
   * pass the output of the prior pipeline to the next pipeline or the query
   * processor
   * in case there are no more pipelines. */
  class DummyPipeline : public Pipeline {
   public:
    DummyPipeline(TablePtr result, const ScanParam& scan_param,
                  PipelineInfoPtr pipe_info);
    const PipelinePtr copy() const;

   protected:
    virtual TablePtr execute_impl();

   private:
    // static const TablePtr do_nothing(const ScanParam& param, StatePtr state);

    DummyPipeline(const DummyPipeline&);
    DummyPipeline& operator=(const DummyPipeline&);
  };

  const TablePtr execute(PipelinePtr pipeline, QueryContextPtr context);

  const PipelinePtr compileQueryFile(const std::string& path_to_query_file,
                                     const ScanParam& param = ScanParam());

  const TablePtr compileAndExecuteQueryFile(
      const std::string& path_to_query_file);

  bool isDummyPipeline(PipelinePtr pipeline);

}  // end namespace CoGaDB

#endif /* PIPELINE_HPP */
