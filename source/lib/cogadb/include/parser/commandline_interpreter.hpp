#pragma once

#include <fstream>
#include <queue>
#include <string>

#include <core/table.hpp>

#include <boost/thread.hpp>
#include <boost/tokenizer.hpp>
#include <parser/client.hpp>

namespace CoGaDB {

  std::string getNextCommandFromGlobalCommandQueue();
  const TablePtr compileAndExecuteQueryFile(
      const std::string& path_to_query_file);

  class CommandLineInterpreter {
   public:
    CommandLineInterpreter(ClientPtr client = ClientPtr(new LocalClient()));

    bool parallelExecution(const std::string& input, std::ifstream* in_stream,
                           std::ofstream* log_file_for_timings);
    void threadedExecution(unsigned int thread_id,
                           std::ofstream* log_file_for_timings,
                           ClientPtr client);
    bool execute(const std::string& command, ClientPtr client);
    bool executeFromFile(const std::string& filepath, ClientPtr client);
    bool getline(const std::string& prompt, std::string& result);
    ClientPtr getClient();

   private:
    typedef bool (*SimpleCommandHandlerPtr)();
    typedef bool (*ParameterizedCommandHandlerPtr)(const std::string&,
                                                   ClientPtr);
    typedef bool (*QueryCommandHandlerPtr)(ClientPtr);
    typedef std::map<std::string, SimpleCommandHandlerPtr> SimpleCommandMap;
    typedef std::map<std::string, QueryCommandHandlerPtr> QueryCommandMap;
    typedef std::map<std::string, ParameterizedCommandHandlerPtr>
        ParameterizedCommandMap;
    SimpleCommandMap simple_command_map_;
    QueryCommandMap query_command_map_;
    ParameterizedCommandMap command_map_;
    std::string prompt_;
    boost::mutex global_logfile_mutex_;
    ClientPtr client_;
  };

}  // end namespace CogaDB
