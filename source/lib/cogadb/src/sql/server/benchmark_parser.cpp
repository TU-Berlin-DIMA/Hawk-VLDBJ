
#include <optimizer/optimizer.hpp>
#include <query_processing/query_processor.hpp>
#include <sql/server/sql_driver.hpp>
#include <util/time_measurement.hpp>

#include <exception>
#include <iostream>
#include <string>

#include <assert.h>

using namespace std;
using namespace CoGaDB;
using namespace query_processing;

SQL::Driver driver;

int main(int argc, char **argv) {
  bool execute = true;

  if (argc > 1 && string(argv[1]) == "--no-execute") {
    execute = false;
    argv[1] = argv[2];
    argc--;
  }

  if (argc != 2) {
    cerr << argv[0] << " [--no-execute] <query>" << endl;
    return 1;
  }

  /* still not every SSB query can be executed on the GPU */
  RuntimeConfiguration::instance().setGlobalDeviceConstraint(
      hype::DeviceConstraint(hype::CPU_ONLY));

  RuntimeConfiguration::instance().setPathToDatabase("./ssb_sf1");
  ClientPtr client = ClientPtr(new LocalClient());
  assert(loadTables(client));

  Timestamp start_ts, parse_ts, execute_ts;

  start_ts = getTimestamp();
  SQL::ParseTree::SequencePtr seq(driver.parse(argv[1]));
  parse_ts = getTimestamp();
  if (execute) {
    seq->execute(client);
    execute_ts = getTimestamp();
  } else {
    execute_ts = start_ts;
    /* total time will be 0 */
  }

  cerr << "Parse: " << (parse_ts - start_ts) << " ns" << endl
       << "Total: " << (execute_ts - start_ts) << " ns" << endl;

  return 0;
}
