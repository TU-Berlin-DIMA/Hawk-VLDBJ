#ifndef PERSEUS_FEATURE_HPP
#define PERSEUS_FEATURE_HPP

#include <memory>
#include <string>

namespace perseus {

  class Feature {
   public:
    virtual const std::string name() const = 0;
    virtual void randomize() = 0;
    virtual void copyValue(const Feature& feature) = 0;
    virtual bool nextValue(signed step = 1) = 0;
    virtual std::unique_ptr<Feature> clone() const = 0;
    virtual const std::string toString() const = 0;
    virtual const unsigned count() const = 0;
    virtual ~Feature() {}
  };
}

#endif  // PERSEUS_FEATURE_HPP
