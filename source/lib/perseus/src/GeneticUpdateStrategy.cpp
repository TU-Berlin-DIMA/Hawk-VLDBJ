#include "GeneticUpdateStrategy.hpp"
#include <utility>
#include "Configuration.hpp"
#include "Feature.hpp"
#include "VariantGenerator.hpp"
#include "utils.h"

namespace perseus {

std::vector<std::unique_ptr<Variant>> GeneticUpdateStrategy::createNewVariants(
    const VariantPool& pool, size_t count,
    const std::vector<std::tuple<double, unsigned>> variantScores) {
  auto matingProbabilities = computeMatingProbabilites(variantScores);
  std::vector<std::unique_ptr<Variant>> variants;
  auto& generator = pool.generator();
  while (variants.size() < count) {
    auto mate1 = findMate(matingProbabilities);
    auto mate2 = findMate(matingProbabilities, mate1);
    auto configuration = createOffspring(pool, mate1, mate2);
    if (generator.validateConfiguration(*configuration)) {
      auto offspring = generator.createVariant(std::move(configuration));
      variants.push_back(std::move(offspring));
    } else {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace) << "Offspring not viable: " << *configuration;
      };
    }
  }
  return variants;
}

const std::vector<std::tuple<double, unsigned>>
GeneticUpdateStrategy::computeMatingProbabilites(
    const std::vector<std::tuple<double, unsigned int>>& variantScores) const {
  std::vector<std::tuple<double, unsigned>> matingProbabilities;
  auto matingProbability = 0.0;
  for (auto& score : variantScores) {
    matingProbability += std::get<0>(score);
    matingProbabilities.push_back(
        std::make_tuple(matingProbability, std::get<1>(score)));
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "Mating score: " << std::get<1>(score) << ": "
                               << matingProbability;
    };
  }
  // fix rounding errors
  std::get<0>(*matingProbabilities.rbegin()) = 1.0;
  return matingProbabilities;
}

const unsigned GeneticUpdateStrategy::findMate(
    const std::vector<std::tuple<double, unsigned>> matingProbabilities,
    const unsigned exclude) {
  double matingProbability = matingDistribution_(global_rnd);
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "Mating probablity: " << matingProbability;
  };
  for (auto& candidate : matingProbabilities) {
    if (std::get<0>(candidate) >= matingProbability) {
      auto index = std::get<1>(candidate);
      if (index == exclude) {
        continue;
      }
      return index;
    }
  }
  // only here to shut up the compiler
  return std::get<1>(*matingProbabilities.rbegin());
}

std::unique_ptr<Configuration> GeneticUpdateStrategy::createOffspring(
    const VariantPool& pool, unsigned mate1, unsigned mate2) {
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "Mating variant " << mate1 << " with " << mate2;
  };
  auto& configuration1 = pool.getVariant(mate1).configuration();
  auto& configuration2 = pool.getVariant(mate2).configuration();
  std::unique_ptr<Configuration> configuration(new Configuration);
  for (auto& feature : configuration1.features()) {
    auto copy = feature->clone();
    if (copy->count() == 1) {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace) << "Passing on feature from variant " << mate1
                                 << ": " << feature->name()
                                 << " (single value feature)";
      };
      copy->copyValue(*feature);
    } else if (matingDistribution_(global_rnd) < mutationProbability_) {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace) << "Mutating feature: " << feature->name();
      };
      copy->randomize();
    } else if (matingDistribution_(global_rnd) < matingProbability_) {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace) << "Passing on feature from variant " << mate1
                                 << ": " << feature->name();
      };
      copy->copyValue(*feature);
    } else {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace) << "Passing on feature from variant " << mate2
                                 << ": " << feature->name();
      };
      copy->copyValue(*configuration2.getFeature(feature->name()));
    }
    configuration->addFeature(std::move(copy));
  }
  return configuration;
}
}