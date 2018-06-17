#ifndef PERSEUS_DISCRETEFEATURE_HPP
#define PERSEUS_DISCRETEFEATURE_HPP

#include <algorithm>
#include <memory>
#include <random>
#include <sstream>
#include <vector>
#include "Feature.hpp"
#include "utils.h"

namespace perseus {

  template <typename T>
  class DiscreteFeature : public Feature {
   private:
    const std::string name_;
    const std::vector<T> values_;
    unsigned index_;
    std::uniform_int_distribution<size_t> distribution;

   public:
    DiscreteFeature(const std::string& name, const std::vector<T>& values)
        : name_(name),
          values_(values),
          index_(0),
          distribution(0, values.size() - 1) {
      assert(values.size() > 0);
    }

    virtual const std::string name() const override { return name_; }

    const std::vector<T> values() const { return values_; }

    const T value() const { return values_[index_]; }

    void setValue(const T value) {
      const auto iterator = std::find(values_.cbegin(), values_.cend(), value);
      if (iterator == values_.cend()) {
        throw std::logic_error(
            "Tried to set discrete feature to invalid value.");
      }
      index_ = iterator - values_.begin();
    }

    virtual void randomize() override {
      if (count() > 1) {
        size_t newIndex = index_;
        while (newIndex == index_) {
          newIndex = distribution(global_rnd);
        }
        index_ = newIndex;
      }
    }

    virtual std::unique_ptr<Feature> clone() const override {
      auto copy = new DiscreteFeature<T>(name_, values_);
      copy->setValue(values_[index_]);
      return std::unique_ptr<Feature>(copy);
    }

    virtual const std::string toString() const override {
      std::stringstream ss;
      ss << name_ << "=" << values_[index_];
      return ss.str();
    }

    virtual void copyValue(const Feature& feature) override {
      const DiscreteFeature<T>& discreteFeature =
          dynamic_cast<const DiscreteFeature<T>&>(feature);
      setValue(discreteFeature.value());
    };

    virtual bool nextValue(signed step = 1) override {
      bool overflow = index_ + step >= count();
      index_ += count() + step;
      index_ %= count();
      return !overflow;
    }

    virtual const unsigned count() const override { return values_.size(); }
  };

  template <typename T>
  const bool operator==(const DiscreteFeature<T>& lhs,
                        const DiscreteFeature<T>& rhs) {
    return lhs.values() == rhs.values() && lhs.value() == rhs.value();
  }
}

#endif  // PERSEUS_DISCRETEFEATURE_HPP
