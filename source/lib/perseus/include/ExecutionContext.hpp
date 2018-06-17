#ifndef PERSEUS_EXECUTIONCONTEXT_HPP
#define PERSEUS_EXECUTIONCONTEXT_HPP

#ifdef __cplusplus

namespace perseus {

  class VariantPool;

  class ExecutionContext {
   public:
    virtual ~ExecutionContext() {}
  };
}

#else   // __cplusplus
typedef struct ExecutionContext ExecutionContext;
#endif  // __cplusplus

#endif  // PERSEUS_EXECUTIONCONTEXT_HPP
