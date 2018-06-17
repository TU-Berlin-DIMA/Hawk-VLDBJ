/*
 * File:   exporter.hpp
 * Author: John Sarrazin
 */
#pragma once
#include <core/table.hpp>
#include <parser/client.hpp>
#include <query_processing/query_processor.hpp>
#include <string>

namespace CoGaDB {

  const std::string IGV_HOST = "localhost";
  // const std::string IGV_PORT = "60151"; //Replaced with Variable in
  // Variable-manager
  const std::string REFERENCE_NAME_QUERY =
      "select rg_name from reference_genome;";
  const std::string CONTIG_NAME_COLUMN = "C_NAME";
  const std::string REFERENCE_BASE_NAME_COLUMN = "RG_NAME";

  // function declarations

  /* \brief This method exports sample genome data to a SAM file.
   *	Command: export_sample_genome <start> <end> [<filename>]
   *	First Parameter specified the beginning of the export
   *	Second Parameter specified the ending of the export
   *	Third optional Parameter specified the Path where to save the SAM file
   */
  bool exportSampleGenome(const std::string& args_string, ClientPtr);

  /* just debug function
   */
  bool debugFunction(const std::string& args_string, ClientPtr);

  // TODO put into test framework
  bool verificateSamFiles(const std::string& args_string, ClientPtr);

}  // end namespace CogaDB
