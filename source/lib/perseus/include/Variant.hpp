#ifndef PERSEUS_VARIANT_HPP
#define PERSEUS_VARIANT_HPP

#include <iostream>

namespace perseus {

  class Configuration;

  class ExecutionContext;

  class Variant {
   public:
    virtual ~Variant() {}
    virtual void invoke(ExecutionContext* context) = 0;
    virtual void waitForLastCall() = 0;
    virtual const unsigned long long totalRuntime() const = 0;
    virtual const unsigned long long totalTuples() const = 0;
    virtual const unsigned long totalCalls() const = 0;
    virtual const double currentRuntimePerTuple() const = 0;
    virtual void setCurrentRuntimePerTuple(double currentRuntimePerTuple) = 0;
    virtual const std::string name() const = 0;
    virtual const Configuration& configuration() const = 0;
    virtual void reset() = 0;
  };

  std::ostream& operator<<(std::ostream& ostream, const Variant& variant);
}

#endif  // PERSEUS_VARIANT_HPP
