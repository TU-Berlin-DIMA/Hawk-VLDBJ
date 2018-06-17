#include <stdio.h>
#include <exception>
#include <iostream>
#include <string>

#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/stream.hpp>

#include "core/variable_manager.hpp"
#include "sql/server/sql_driver.hpp"
#include "sql/server/sql_parsetree.hpp"
#include "sql_parser.hpp"

#include <optimizer/optimizer.hpp>
#include <util/query_processing.hpp>

using namespace boost;

namespace CoGaDB {
namespace SQL {

Driver::Driver() : buffer(NULL), istream(NULL), parser(*this, scanner) {
  init_scan();
  //	parser.set_debug_level(1);
}

ParseTree::SequencePtr Driver::parse(std::istream &is) {
  set_input(is);
  if (parser.parse()) throw ParseError();

  return result;
}

ParseTree::SequencePtr Driver::parse(const std::string &src) {
  set_input(src);
  if (parser.parse()) throw ParseError();

  return result;
}

Driver::~Driver() { destroy_scan(); }

bool commandlineExec(const std::string &input, ClientPtr client) {
  Timestamp begin = getTimestamp();

  TablePtr result = executeSQL(input, client);

  Timestamp end = getTimestamp();

  assert(end >= begin);
  double exec_time_in_milliseconds = double(end - begin) / (1000 * 1000);
  return printResult(result, client, exec_time_in_milliseconds);
}

TablePtr executeSQL(const std::string &input, ClientPtr client) {
  Driver driver;
  ParseTree::SequencePtr seq;
  TablePtr result;

  std::ostream &out = client->getOutputStream();

  try {
    seq = driver.parse(input);
    result = seq->execute(client);

  } catch (const std::exception &e) {
    out << e.what() << std::endl;
    return TablePtr();
  }

  return result;
}

bool commandlineExplain(const std::string &input, ClientPtr client) {
  // mark as unused
  (void)client;
  Driver driver;
  ParseTree::SequencePtr seq;

  try {
    seq = driver.parse(input);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }

  /*
   * FIXME: this only works on POSIX
   */
  FILE *pipe = popen("dot -Tgtk", "w");
  if (!pipe) {
    perror("Error opening `dot`");
    return false;
  }
  iostreams::file_descriptor_sink fd(fileno(pipe), iostreams::close_handle);
  iostreams::stream<iostreams::file_descriptor_sink> stream(fd);

  seq->explain(stream);

  pclose(pipe);
  return true;
}

bool commandlineExplainStatements(const std::string &input, ClientPtr client,
                                  bool optimize) {
  std::ostream &out = client->getOutputStream();
  Driver driver;
  ParseTree::SequencePtr seq;

  try {
    seq = driver.parse(input);
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return false;
  }
  std::list<CoGaDB::query_processing::LogicalQueryPlanPtr> plans =
      seq->getLogicalQueryPlans();
  std::list<CoGaDB::query_processing::LogicalQueryPlanPtr>::iterator it;
  unsigned int i = 0;
  for (it = plans.begin(); it != plans.end(); ++it) {
    if (plans.size() > 1)
      out << "Query Plan for Statement " << i++ << " : " << std::endl;
    query_processing::LogicalQueryPlanPtr plan = *it;
    if (plan) {
      plan->setOutputStream(out);
      if (!optimizer::checkQueryPlan(plan->getRoot())) {
        client->getOutputStream() << "Query Compilation Failed!" << std::endl;
        return false;
      }
      if (optimize) {
        optimizer::Logical_Optimizer::instance().optimize(plan);
      }
      plan->print();
    }
  }
  return true;
}

bool commandlineExplainStatementsWithoutOptimization(const std::string &input,
                                                     ClientPtr client) {
  return commandlineExplainStatements(input, client, false);
}

bool commandlineExplainStatementsWithOptimization(const std::string &input,
                                                  ClientPtr client) {
  return commandlineExplainStatements(input, client, true);
}

} /* namespace SQL */
} /* namespace CoGaDB */
