#include <fstream>
#include <iostream>

#include <dlfcn.h>
#include <stdlib.h>

#include <core/global_definitions.hpp>

#include <persistence/storage_manager.hpp>
#include <query_compilation/code_generators/cpp_code_generator.hpp>

#include <core/selection_expression.hpp>
#include <parser/commandline_interpreter.hpp>
#include <util/time_measurement.hpp>

#include <boost/make_shared.hpp>
#include <iomanip>
#include <util/getname.hpp>

#include <boost/program_options.hpp>

#include <util/tests.hpp>

#include "core/variable_manager.hpp"
#include "query_compilation/code_generators/code_generator_utils.hpp"

using namespace CoGaDB;

const TablePtr qcrev_tpch1(CodeGeneratorType code_generator);
const TablePtr qcrev_tpch5_join(CodeGeneratorType code_generator);
const TablePtr qcrev_tpch9(CodeGeneratorType code_generator);
const TablePtr qcrev_tpch13(CodeGeneratorType code_generator);
const TablePtr qcrev_tpch17(CodeGeneratorType code_generator);
const TablePtr qcrev_tpch18(CodeGeneratorType code_generator);
const TablePtr qcrev_tpch19(CodeGeneratorType code_generator);
const TablePtr qcrev_tpch21(CodeGeneratorType code_generator);
