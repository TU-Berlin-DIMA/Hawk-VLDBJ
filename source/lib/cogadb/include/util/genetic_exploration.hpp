#ifndef GENETIC_EXPLORATION_HPP
#define GENETIC_EXPLORATION_HPP

#include <query_compilation/code_generator.hpp>
#include <query_processing/definitions.hpp>
#include <util/variant_configurator.hpp>

namespace CoGaDB {

  Variant genetic_exploration(VariantIterator iterator,
                              query_processing::LogicalQueryPlanPtr log_plan,
                              CodeGeneratorType code_gen_type,
                              ClientPtr client);
}

#endif  // GENETIC_EXPLORATION_HPP
