#include <util/genetic_exploration.hpp>

#include <util/query_processing.hpp>

#include <limits>
#include <random>

namespace CoGaDB {

class FitnessStore {
 public:
  double getFitness(const Variant& var) const {
    auto find = fitness_store_.find(var);

    if (find != fitness_store_.end()) {
      return find->second;
    } else {
      return std::numeric_limits<double>::max();
    }
  }

  void add(const Variant& var, query_processing::LogicalQueryPlanPtr log_plan,
           CodeGeneratorType code_gen_type, ClientPtr client) {
    const auto number_of_runs = 5u;
    auto exploration_max_pipe_exec_time_in_s =
        VariableManager::instance().getVariableValueInteger(
            "bench.exploration.max_pipe_exec_time_in_s");
    double fitness = 0;
    auto i = 0u;

    for (; i < number_of_runs; ++i) {
      auto ret =
          executeQueryPlanWithCompiler(log_plan, client, code_gen_type, &var);

      fitness += ret.second.total_pipeline_execution_time_in_s;

      if (ret.second.total_pipeline_execution_time_in_s >
          exploration_max_pipe_exec_time_in_s) {
        break;
      }

      fitness_store_[var] = fitness / i;
    }

    fitness_store_[var] =
        executeQueryPlanWithCompiler(log_plan, client, code_gen_type, &var)
            .second.total_pipeline_execution_time_in_s;
  }

  bool hasFitness(const Variant& var) const {
    return fitness_store_.find(var) != fitness_store_.end();
  }

 private:
  std::map<Variant, double> fitness_store_;
};

std::set<uint32_t> getRandomIndices(const uint32_t count,
                                    const uint32_t max_index) {
  std::set<uint32_t> choosen;
  std::random_device random_device;
  std::mt19937 engine(random_device());
  std::uniform_int_distribution<uint32_t> dist(0, max_index);

  while (choosen.size() != count) {
    choosen.insert(dist(engine));
  }

  return choosen;
}

using VariantMap = std::map<std::string, std::vector<std::string>>;

Variant crossover(const Variant& var0, const Variant& var1) {
  const auto selection = 0.5;
  Variant new_var;

  std::random_device random_device;
  std::mt19937 engine(random_device());
  std::uniform_real_distribution<> dist(0, 1);

  for (const auto& dim : var0) {
    if (dist(engine) < selection) {
      new_var.insert(dim);
    } else {
      new_var.insert(*var1.find(dim.first));
    }
  }

  return new_var;
}

void mutate_variant(Variant& var, const float mutation_rate,
                    const VariantMap& var_map) {
  std::random_device random_device;
  std::mt19937 engine(random_device());
  std::uniform_real_distribution<> dist(0, 1);

  for (auto& dim : var) {
    if (dist(engine) < mutation_rate) {
      const auto& values = var_map.find(dim.first)->second;

      dim.second = *std::next(values.begin(), std::rand() % values.size());
    }
  }
}

class Population {
 public:
  Population(FitnessStore* fitness_store) : fitness_store_(fitness_store) {}

  void add(const Variant& var) { population_.push_back(var); }

  Variant getFittest() const {
    Variant fittest;
    double fittest_fitness = std::numeric_limits<double>::max();
    for (const auto& var : population_) {
      auto fitness = fitness_store_->getFitness(var);
      if (fitness < fittest_fitness) {
        fittest = var;
        fittest_fitness = fitness;
      }
    }

    return fittest;
  }

  double getFittestFitness() const {
    return fitness_store_->getFitness(getFittest());
  }

  Population evolve(bool elitism, float mutation_rate,
                    const VariantMap& variant_map,
                    query_processing::LogicalQueryPlanPtr log_plan,
                    CodeGeneratorType code_gen_type, ClientPtr client) {
    Population new_pop(fitness_store_);

    if (elitism) {
      new_pop.add(getFittest());
    }

    for (auto i = elitism ? 1u : 0u; i < population_.size(); ++i) {
      auto var0 = tournamentSelection();
      auto var1 = tournamentSelection();
      auto new_var = crossover(var0, var1);

      if (!fitness_store_->hasFitness(new_var)) {
        fitness_store_->add(new_var, log_plan, code_gen_type, client);
      }

      new_pop.add(new_var);
    }

    new_pop.mutate(mutation_rate, variant_map, log_plan, code_gen_type, client);

    return new_pop;
  }

 private:
  Variant tournamentSelection() {
    const uint32_t tournament_size = 5;
    auto indices = getRandomIndices(tournament_size, population_.size() - 1);

    Variant fittest;
    double fittest_fitness = std::numeric_limits<double>::max();

    for (auto index : indices) {
      auto variant = *std::next(population_.begin(), index);
      auto fitness = fitness_store_->getFitness(variant);

      if (fitness < fittest_fitness) {
        fittest = variant;
        fittest_fitness = fitness;
      }
    }

    return fittest;
  }

  void mutate(float mutation_rate, const VariantMap& variant_map,
              query_processing::LogicalQueryPlanPtr log_plan,
              CodeGeneratorType code_gen_type, ClientPtr client) {
    for (auto& var : population_) {
      mutate_variant(var, mutation_rate, variant_map);

      if (!fitness_store_->hasFitness(var)) {
        fitness_store_->add(var, log_plan, code_gen_type, client);
      }
    }
  }

  std::vector<Variant> population_;
  FitnessStore* fitness_store_;
};

Population createInitialPopulation(
    FitnessStore* fitness_store, const VariantMap& variant_map, uint32_t size,
    query_processing::LogicalQueryPlanPtr log_plan,
    CodeGeneratorType code_gen_type, ClientPtr client) {
  Population population(fitness_store);

  std::random_device random_device;
  std::mt19937 engine(random_device());
  std::uniform_int_distribution<uint32_t> dist;

  std::set<Variant> variants;

  while (variants.size() < size) {
    Variant var;

    for (const auto& dim : variant_map) {
      var.insert(std::make_pair(dim.first,
                                dim.second[dist(engine) % dim.second.size()]));
    }

    variants.insert(var);
  }

  for (const auto& var : variants) {
    fitness_store->add(var, log_plan, code_gen_type, client);
    population.add(var);
  }

  return population;
}

VariantMap generateVariantMap(VariantIterator& itr) {
  VariantMap result;

  for (const auto& var : itr.getFlattenedDimensions()) {
    result.insert(var);
  }

  return result;
}

bool compareFittest(const std::vector<Variant>& fittest,
                    const uint32_t fittest_to_check) {
  if (fittest.size() < fittest_to_check) {
    return false;
  }

  const auto& to_check = fittest.back();

  uint32_t checked = 0;
  for (auto itr(fittest.rbegin()), end(fittest.rend());
       itr != end && checked < fittest_to_check; ++itr, ++checked) {
    if (*itr != to_check) {
      return false;
    }
  }

  return true;
}

Variant genetic_exploration(VariantIterator iterator,
                            query_processing::LogicalQueryPlanPtr log_plan,
                            CodeGeneratorType code_gen_type, ClientPtr client) {
  const float mutation_rate = 0.15f;
  const uint32_t number_of_fittest_to_check = 10;
  const uint32_t population_size =
      VariableManager::instance().getVariableValueInteger(
          "code_gen.genetic_exploration.population_size");

  const bool elitism = VariableManager::instance().getVariableValueBoolean(
      "code_gen.genetic_exploration.elitism");

  uint32_t max_iteration = VariableManager::instance().getVariableValueInteger(
      "code_gen.genetic_exploration.max_iteration_count");

  auto variant_map = generateVariantMap(iterator);
  FitnessStore fitness_store;
  auto population =
      createInitialPopulation(&fitness_store, variant_map, population_size,
                              log_plan, code_gen_type, client);

  std::vector<Variant> fittest;

  for (auto i = 0u; i < max_iteration; ++i) {
    fittest.push_back(population.getFittest());

    if (compareFittest(fittest, number_of_fittest_to_check)) {
      break;
    }

    population = population.evolve(elitism, mutation_rate, variant_map,
                                   log_plan, code_gen_type, client);
  }

  auto fittest_variant = population.getFittest();

  client->getOutputStream()
      << "FittestVariantRuntime: " << population.getFittestFitness() << "s"
      << std::endl;
  client->getOutputStream() << "FittestVariant:" << std::endl;
  print(client, fittest_variant);

  return fittest_variant;
}
}
