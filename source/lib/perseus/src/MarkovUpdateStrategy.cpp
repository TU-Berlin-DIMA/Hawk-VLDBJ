#include "MarkovUpdateStrategy.hpp"
#include <cassert>
#include <random>
#include "Configuration.hpp"
#include "utils.h"

namespace perseus {

void MarkovUpdateStrategy::updatePool(VariantPool& pool) {
  assert(pool.poolSize() % 2 == 0);
  assert(pool.minimumSize() % 2 == 0);
  auto scores = variantScorer_->scoreVariants(pool.variants());
  auto newPoolSize = computeNewPoolSize(pool);
  auto chainsToRemove = computeChainsToRemove(pool.poolSize(), newPoolSize);
  auto chainsToKeep = computeChainsToKeep(newPoolSize, scores.size());
  determineVariantsToKeep(pool, scores);
  removeChains(pool, chainsToRemove, scores);
  createNewSuccessors(pool, chainsToKeep);
}

const size_t MarkovUpdateStrategy::computeNewPoolSize(
    const VariantPool& pool) const {
  auto newPoolSize = std::max(static_cast<size_t>(std::lround(
                                  pool.poolSize() * pool.reductionFactor())),
                              pool.minimumSize());
  newPoolSize -= (newPoolSize % 2);
  return newPoolSize;
}

const size_t MarkovUpdateStrategy::computeChainsToKeep(
    const size_t newPoolSize, const size_t testedVariants) const {
  return std::min(newPoolSize, testedVariants) / 2;
}

const size_t MarkovUpdateStrategy::computeChainsToRemove(
    const size_t poolSize, const size_t newPoolSize) const {
  return (poolSize - newPoolSize) / 2;
}

void MarkovUpdateStrategy::removeChains(
    VariantPool& pool, const size_t count,
    const std::vector<std::tuple<double, unsigned>>& scores) {
  if (count == 0) {
    return;
  }
  std::vector<std::tuple<double, unsigned>> chainScores;
  for (auto i = 0u; i < scores.size() / 2; ++i) {
    auto chainScore =
        std::min(std::get<0>(scores[2 * i]), std::get<0>(scores[2 * i + 1]));
    chainScores.push_back(std::make_tuple(chainScore, i));
  }
  std::sort(chainScores.begin(), chainScores.end(),
            std::less<std::tuple<double, unsigned>>());
  for (auto i = 0u; i < count; ++i) {
    auto chain = std::get<1>(chainScores[i]);
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "Removing variants in chain " << chain;
    };
    pool.removeVariant(2 * chain + 1);
    pool.removeVariant(2 * chain);
  }
}

void MarkovUpdateStrategy::determineVariantsToKeep(
    VariantPool& pool,
    const std::vector<std::tuple<double, unsigned>>& scores) {
  auto chains = scores.size() / 2;
  PERSEUS_TRACE {
    auto untested = pool.poolSize() / 2 - chains;
    if (untested > 0) {
      BOOST_LOG_TRIVIAL(trace) << "Ignoring " << untested
                               << " untested chains.";
    }
  };
  for (auto i = 0u; i < chains; ++i) {
    auto score1 = std::get<0>(scores[2 * i]);
    auto score2 = std::get<0>(scores[2 * i + 1]);
    auto probability = score2 / (score1 + score2);
    auto draw = selectionDistribution(global_rnd);
    auto keepSuccessor = probability >= draw;
    if (keepSuccessor) {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace)
            << "Variant " << i * 2 << " score: " << score1 << "; variant "
            << i * 2 + 1 << " score: " << score2
            << "; chance to keep successor: " << probability
            << "; draw: " << draw << "; keeping successor in chain " << i;
      };
      pool.swapVariants(2 * i, 2 * i + 1);
    } else {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace)
            << "Variant " << i * 2 << " score: " << score1 << "; variant "
            << i * 2 + 1 << " score: " << score2
            << "; chance to keep successor: " << probability
            << "; draw: " << draw << "; keeping predecessor in chain " << i;
      }
    }
  }
}

void MarkovUpdateStrategy::createNewSuccessors(VariantPool& pool,
                                               const size_t chains) {
  auto& generator = pool.generator();
  for (auto i = 0u; i < chains; ++i) {
    auto configuration = createSuccessor(pool, i);
    while (!generator.validateConfiguration(*configuration)) {
      PERSEUS_TRACE {
        BOOST_LOG_TRIVIAL(trace) << "Successor not viable: " << *configuration;
      };
      configuration = createSuccessor(pool, i);
    }
    auto successor = generator.createVariant(std::move(configuration));
    PERSEUS_TRACE {
      BOOST_LOG_TRIVIAL(trace) << "Swapping " << pool.getVariant(2 * i + 1)
                               << " with " << *successor;
    };
    pool.updateVariant(2 * i + 1, std::move(successor));
  }
}

std::unique_ptr<Configuration> MarkovUpdateStrategy::createSuccessor(
    const VariantPool& pool, const unsigned chain) const {
  auto& predecessor = pool.getVariant(2 * chain);
  auto copy = new Configuration(predecessor.configuration());
  std::unique_ptr<Configuration> configuration(copy);
  auto features = configuration->features();
  std::uniform_int_distribution<unsigned> featureRandomizationDistribution(
      0, features.size() - 1);
  Feature* feature = nullptr;
  while (feature == nullptr) {
    auto index = featureRandomizationDistribution(global_rnd);
    feature = features[index];
    if (feature->count() == 1) {
      feature = nullptr;
    }
  }
  PERSEUS_TRACE {
    BOOST_LOG_TRIVIAL(trace) << "Updating feature: " << feature->name();
  };
  feature->randomize();
  return configuration;
}
}