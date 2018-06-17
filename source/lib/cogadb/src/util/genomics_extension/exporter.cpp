/*
 * File:   exporter.cpp
 * Author: John Sarrazin
 */

#include <boost/algorithm/string.hpp>
#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <util/genomics_extension/exporter.hpp>

#include <compression/rle_compressed_column.hpp>
#include <core/table.hpp>
#include "core/variable_manager.hpp"
#include "sql/server/sql_driver.hpp"
#include "util/genomics_extension/CigarFSM.hpp"
#include "util/genomics_extension/samExportCommandParser.hpp"
#include "util/genomics_extension/samQueryCreator.hpp"
#include "util/genomics_extension/samVerification.hpp"
#include "util/time_measurement.hpp"

namespace CoGaDB {

// forward declaration
bool _exportDatabaseToCSV(ClientPtr client);

bool _igvControl(const std::string ip, const std::string port,
                 const std::string refGenomeId,
                 const std::string sampleGenomePath, const std::string locus);

bool _igvControl(const std::string ip, const std::string port,
                 const std::string refGenomeId,
                 const std::string sampleGenomePath, const std::string locus,
                 std::string snapshotDirectory, std::string snapshotName);
std::string _calculateSamfile(TablePtr cigarBaseTable);

/**
 * \brief Entry Point for the SamExporter. Exports a Sam-File based on the
 * Parameter
 * Control-Function for the Sam Export functionality.
 */
bool exportSampleGenome(const std::string &args_string, ClientPtr client) {
  using std::endl;
  std::ostream &out = client->getOutputStream();

  // Timekeeping
  Timestamp totalTimeStart = getTimestamp();
  Timestamp localTimeStart = getTimestamp();
  bool timeKeepingEnabled = false;

  // Parse in Arguments
  std::vector<std::string> args;
  boost::split(args, args_string, boost::is_any_of(" "));
  boost::shared_ptr<SamExportParser> samExportParser;
  std::string exporterInputType =
      VariableManager::instance().getVariableValueString(
          "sam_exporter_input_api");
  if (exporterInputType == "simple") {
    samExportParser =
        boost::shared_ptr<SamExportParser>(new SimpleSamExportParser());
  } else {
    out << "Error: Input Api " << exporterInputType << " is not defined."
        << endl;
    return false;
  }
  if (!samExportParser->addArguments(args, out)) {
    out << "Error: Parsing arguments failed. Simple Input Api for Sam Export: "
        << CoGaDB::SAM_EXPORTER_SIMPLE_INPUT_API_DEFINITION << "." << endl;
    return false;
  }

  timeKeepingEnabled = samExportParser->isTimeKeepingEnabled();

  // Build Query
  SamQueryCreator queryBuilder(samExportParser->getSamSelection());

  if (timeKeepingEnabled) {
    out << "Parsing Arguments: "
        << (double(getTimestamp() - localTimeStart)) / 1000000000 << "s"
        << endl;
    localTimeStart = getTimestamp();
  }

  // Execute Query
  TablePtr samCalcBaseTable =
      CoGaDB::SQL::executeSQL(queryBuilder.getQuery(), client);

  if (timeKeepingEnabled) {
    out << "Executing SQL-Query: "
        << (double(getTimestamp() - localTimeStart)) / 1000000000 << "s"
        << endl;
    localTimeStart = getTimestamp();
  }

  if (samCalcBaseTable->getNumberofRows() == 0) {
    out << "Error: No data selected." << endl;
    return false;
  }

  // Generate Samfile
  // std::string tmpSamOutput = samCalcBaseTable->toString("sam"); //old version
  // over result_output_format
  std::string tmpSamOutput = _calculateSamfile(samCalcBaseTable);
  if (timeKeepingEnabled) {
    out << "Calculating Samfile: "
        << (double(getTimestamp() - localTimeStart)) / 1000000000 << "s"
        << endl;
    localTimeStart = getTimestamp();
  }

  // Write into file
  std::string samSavePath = samExportParser->getSavePath();
  std::ofstream file(samSavePath.c_str(), std::ios::trunc);
  if (!file.good()) {
    out << "Could not open file: '" << samSavePath << "'" << endl;
    return false;
  }
  file << tmpSamOutput;
  file.close();
  if (timeKeepingEnabled) {
    out << "Writing Data into File: "
        << (double(getTimestamp() - localTimeStart)) / 1000000000 << "s"
        << endl;
    out << "Total Time:  "
        << (double(getTimestamp() - totalTimeStart)) / 1000000000 << "s"
        << endl;
  }
  // Control IGV
  if (samExportParser->isIgvControlEnabled()) {
    // delete old index files
    try {
      std::string oldIndexPath = samSavePath;
      oldIndexPath += ".sai";
      boost::filesystem::remove(oldIndexPath);
    } catch (boost::filesystem::filesystem_error &e) {
      out << "Error. Could not delete old index file. " << e.what()
          << std::endl;
    }

    // index samfile if igvtools is avaiable
    std::string indexCommand = "./IGVTools/igvtools index ";
    indexCommand += samSavePath;
    indexCommand += " >>null";  // Ausgabe ins nichts umleiten

    if (timeKeepingEnabled) localTimeStart = getTimestamp();

    int ret = system(indexCommand.c_str());
    if (ret) {
      std::cerr << "Indexing command failed!" << std::endl;
      return false;
    }

    if (timeKeepingEnabled)
      out << "Indexing Sam-File: "
          << (double(getTimestamp() - localTimeStart)) / 1000000000 << "s"
          << endl;

    // igvLocus = <first_contig>:<start>-<end>
    std::stringstream igvLocusStringStream;
    std::ostream &igvLocusOut = igvLocusStringStream;
    int start = samExportParser->getStartValue();
    int end = samExportParser->getEndValue();
    if (start <= 0) start = 1;

    igvLocusOut << samCalcBaseTable->getColumnbyName(CONTIG_NAME_COLUMN)
                       ->getStringValue(0);
    if (end >= 0) {
      igvLocusOut << ":" << boost::lexical_cast<std::string>(start) << "-"
                  << boost::lexical_cast<std::string>(end);
    }
    // TablePtr referenceNameTable =
    // CoGaDB::SQL::executeSQL(REFERENCE_NAME_QUERY, client);
    if (samExportParser->isSnapshotEnabled()) {
      if (!_igvControl(
              IGV_HOST,
              VariableManager::instance().getVariableValueString("igv_port"),
              samCalcBaseTable->getColumnbyName(REFERENCE_BASE_NAME_COLUMN)
                  ->getStringValue(0),
              samSavePath, igvLocusStringStream.str(),
              VariableManager::instance().getVariableValueString(
                  "igv_snapshot_path"),
              VariableManager::instance().getVariableValueString(
                  "igv_snapshot_name"))) {
        out << "Error: IGV-Control failed." << endl;
      }
    } else {
      if (!_igvControl(
              IGV_HOST,
              VariableManager::instance().getVariableValueString("igv_port"),
              samCalcBaseTable->getColumnbyName(REFERENCE_BASE_NAME_COLUMN)
                  ->getStringValue(0),
              samSavePath, igvLocusStringStream.str())) {
        out << "Error: IGV-Control failed." << endl;
      }
    }
  }
  return true;
}

bool igvControl(const std::string &args_string, ClientPtr client) {
  using std::endl;
  std::ostream &out = client->getOutputStream();
  out << "DEBUG: Not yet implemented!" << endl;
  return false;
}

inline bool _igvControl(const std::string ip, const std::string port,
                        const std::string refGenomeId,
                        const std::string sampleGenomePath,
                        const std::string locus) {
  using std::endl;
  using boost::asio::ip::tcp;

  try {
    boost::asio::io_service io_service;

    // Get a list of endpoints corresponding to the server name.
    tcp::resolver resolver(io_service);
    tcp::resolver::query query(ip.c_str(), port.c_str());
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

    // Try each endpoint until we successfully establish a connection.
    tcp::socket socket(io_service);
    boost::asio::connect(socket, endpoint_iterator);

    boost::asio::streambuf request;
    std::ostream request_stream(&request);

    boost::asio::write(socket, request);
    boost::asio::streambuf response;
    boost::system::error_code error;

    request_stream << "new" << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    request_stream << "genome " << refGenomeId << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    request_stream << "load " << sampleGenomePath << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    request_stream << "goto " << locus << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    // Closing Socket
    socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
    socket.close();
  } catch (std::exception &e) {
    std::cout << e.what() << "\n";
    return false;
  }
  return true;
}

inline bool _igvControl(const std::string ip, const std::string port,
                        const std::string refGenomeId,
                        const std::string sampleGenomePath, std::string locus,
                        std::string snapshotDirectory,
                        std::string snapshotName) {
  using std::endl;
  using boost::asio::ip::tcp;

  try {
    boost::asio::io_service io_service;

    // Get a list of endpoints corresponding to the server name.
    tcp::resolver resolver(io_service);
    tcp::resolver::query query(ip.c_str(), port.c_str());
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

    // Try each endpoint until we successfully establish a connection.
    tcp::socket socket(io_service);
    boost::asio::connect(socket, endpoint_iterator);

    boost::asio::streambuf request;
    std::ostream request_stream(&request);

    boost::asio::write(socket, request);
    boost::asio::streambuf response;
    boost::system::error_code error;

    request_stream << "new" << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    request_stream << "genome " << refGenomeId << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    request_stream << "load " << sampleGenomePath << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    request_stream << "goto " << locus << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    request_stream << "snapshotDirectory " << snapshotDirectory << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    request_stream << "snapshot " << snapshotName << endl;
    boost::asio::write(socket, request);
    boost::asio::read_until(socket, response, "OK");

    socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
    socket.close();
  } catch (std::exception &e) {
    std::cout << "Exception: " << e.what() << "\n";
    return false;
  }
  return true;
}

/**
 * \brief Just a debug function to test some shit
 */
bool debugFunction(const std::string &args_string, ClientPtr client) {
  using std::endl;

  return true;
}

/**
 * TODO make a function to use this as Unit-test
 */
bool verificateSamFiles(const std::string &args_string, ClientPtr client) {
  using std::endl;
  std::ostream &out = client->getOutputStream();
  std::vector<std::string> args;
  boost::split(args, args_string, boost::is_any_of(" "));
  if (args.size() != 2) {
    out << "Error: Not enough Parameter." << endl;
    return false;
  }
  // return
  // CoGaDB::compareSamFiles("/home/john/coga_samfiles_csvfiles/Harrington_contig1-30.sam","/home/john/coga_samfiles_csvfiles/default.sam",
  // out);
  CoGaDB::compareSamFiles(args[0], args[1], out);

  return true;
}

bool _exportDatabaseToCSV(ClientPtr client) {
  using std::endl;
  std::ostream &out = client->getOutputStream();

  // Export Database for bug fixing
  out << "DEBUG: This is a Debug function to export some Tables to csv."
      << endl;

  // DEBUG
  // Sample_base
  TablePtr cigarBaseTable =
      CoGaDB::SQL::executeSQL("select * from sample_base;", client);
  // std::string samOutput = cigarBaseTable->toString("sam");
  std::ofstream file1("sample_base.csv", std::ios::trunc);
  if (!file1.good()) {
    return false;
  }
  file1 << cigarBaseTable->toString("csv");
  file1.close();

  // reference_base     order by sb_read_id, rb_position, sb_insert_offset;
  cigarBaseTable = CoGaDB::SQL::executeSQL(
      "select * from reference_base order by rb_position;", client);
  // std::string samOutput = cigarBaseTable->toString("sam");
  std::ofstream file2("reference_base.csv", std::ios::trunc);
  if (!file2.good()) {
    return false;
  }
  file2 << cigarBaseTable->toString("csv");
  file2.close();

  // contig
  cigarBaseTable = CoGaDB::SQL::executeSQL("select * from contig;", client);
  // std::string samOutput = cigarBaseTable->toString("sam");
  std::ofstream file3("contig.csv", std::ios::trunc);
  if (!file3.good()) {
    return false;
  }
  file3 << cigarBaseTable->toString("csv");
  file3.close();

  // read     order by sb_read_id, rb_position, sb_insert_offset;
  cigarBaseTable = CoGaDB::SQL::executeSQL("select * from read;", client);
  // std::string samOutput = cigarBaseTable->toString("sam");
  std::ofstream file4("read.csv", std::ios::trunc);
  if (!file4.good()) {
    return false;
  }
  file4 << cigarBaseTable->toString("csv");
  file4.close();

  // samplebase join reference base     order by sb_read_id, rb_position,
  // sb_insert_offset;
  cigarBaseTable = CoGaDB::SQL::executeSQL(
      "select * from sample_base join reference_base on sb_rb_id = rb_id order "
      "by sb_read_id, rb_position, sb_insert_offset;",
      client);
  // std::string samOutput = cigarBaseTable->toString("sam");
  std::ofstream file5("sampleJoinReference.csv", std::ios::trunc);
  if (!file5.good()) {
    return false;
  }
  file5 << cigarBaseTable->toString("csv");
  file5.close();

  // samplebase join reference base join contig
  cigarBaseTable = CoGaDB::SQL::executeSQL(
      "select * from sample_base join reference_base on sb_rb_id = rb_id join "
      "contig on rb_c_id = c_id order by sb_read_id, rb_position, "
      "sb_insert_offset;",
      client);
  // std::string samOutput = cigarBaseTable->toString("sam");
  std::ofstream file6("sampleJoinReferenceJoinContig.csv", std::ios::trunc);
  if (!file6.good()) {
    return false;
  }
  file6 << cigarBaseTable->toString("csv");
  file6.close();

  // samplebase join reference base join contig join read
  cigarBaseTable = CoGaDB::SQL::executeSQL(
      "select * from sample_base join reference_base on sb_rb_id = rb_id join "
      "contig on rb_c_id = c_id join read on sb_read_id = r_id order by "
      "sb_read_id, rb_position, sb_insert_offset;",
      client);
  // std::string samOutput = cigarBaseTable->toString("sam");
  std::ofstream file7("sampleJoinReferenceJoinContigJoinRead.csv",
                      std::ios::trunc);
  if (!file7.good()) {
    return false;
  }
  file7 << cigarBaseTable->toString("csv");
  file7.close();

  return true;
}

std::string _calculateSamfile(TablePtr cigarBaseTable) {
  using std::endl;

  typedef ColumnBaseTyped<std::string> StringTypedColumnType;
  typedef boost::shared_ptr<StringTypedColumnType> StringTypedColumnPtr;
  typedef ColumnBaseTyped<TID> OIDTypedColumnType;
  typedef boost::shared_ptr<OIDTypedColumnType> OIDTypedColumnPtr;
  typedef ColumnBaseTyped<int> IntTypedColumnType;
  typedef boost::shared_ptr<IntTypedColumnType> IntTypedColumnPtr;

  // Saving Pointer to needed Columns
  OIDTypedColumnPtr sb_read_id =
      boost::dynamic_pointer_cast<OIDTypedColumnType>(
          cigarBaseTable->getColumnbyName("SB_READ_ID"));  // 0
  assert(sb_read_id != NULL);
  StringTypedColumnPtr r_qname =
      boost::dynamic_pointer_cast<StringTypedColumnType>(
          cigarBaseTable->getColumnbyName("R_QNAME"));  // 1
  assert(r_qname != NULL);
  IntTypedColumnPtr r_flag = boost::dynamic_pointer_cast<IntTypedColumnType>(
      cigarBaseTable->getColumnbyName("R_FLAG"));  // 2
  assert(r_flag != NULL);
  StringTypedColumnPtr c_name =
      boost::dynamic_pointer_cast<StringTypedColumnType>(
          cigarBaseTable->getColumnbyName("C_NAME"));  // 3
  assert(c_name != NULL);
  IntTypedColumnPtr rb_position =
      boost::dynamic_pointer_cast<IntTypedColumnType>(
          cigarBaseTable->getColumnbyName("RB_POSITION"));  // 4
  assert(rb_position != NULL);
  IntTypedColumnPtr r_mapq = boost::dynamic_pointer_cast<IntTypedColumnType>(
      cigarBaseTable->getColumnbyName("R_MAPQ"));  // 5
  assert(r_mapq != NULL);
  StringTypedColumnPtr sb_base_value =
      boost::dynamic_pointer_cast<StringTypedColumnType>(
          cigarBaseTable->getColumnbyName("SB_BASE_VALUE"));  // 6
  assert(sb_base_value != NULL);
  IntTypedColumnPtr sb_insert_offset =
      boost::dynamic_pointer_cast<IntTypedColumnType>(
          cigarBaseTable->getColumnbyName("SB_INSERT_OFFSET"));  // 7
  assert(sb_insert_offset != NULL);

  // boost::shared_ptr<LookupArray<int> > sb_insert_offset_lookup =
  //        boost::dynamic_pointer_cast<LookupArray<int> > (sb_insert_offset);
  // assert(sb_insert_offset_lookup != NULL);

  // boost::shared_ptr<RLECompressedColumn<int> > sb_insert_offset_rle =
  //        boost::dynamic_pointer_cast<RLECompressedColumn<int>
  //        >(sb_insert_offset_lookup->getIndexedColumn());//7
  // PositionListPtr sb_insert_offset_tids =
  // sb_insert_offset_lookup->getPositionList();
  // assert(sb_insert_offset_rle != NULL);

  IntTypedColumnPtr sb_base_call_quality =
      boost::dynamic_pointer_cast<IntTypedColumnType>(
          cigarBaseTable->getColumnbyName("SB_BASE_CALL_QUALITY"));  // 8

  // Getting output Header
  std::stringstream ss;
  std::ostream &out = ss;
  std::stringstream headerStringStream;
  std::ostream &headerOutStream = headerStringStream;

  // Initialisation
  int currentReadId = -1;
  std::string currentQName("");
  int currentFlag;
  std::string currentCname("");
  unsigned int currentSmallestRbPosition(0);
  int currentMapQ;
  std::string currentBaseValueSequence("");
  std::string currentCigarString("");
  std::string currentQual("");

  std::string currentSamLine("");

  CigarFSM cigarFSM;

  headerOutStream << "@HD\t"
                  << "VN:1.0\t"
                  << "SO:coordinate" << endl;

  // Row Iteration
  for (unsigned int i = 0; i < sb_read_id->size(); i++) {
    int tmpReadId;
    tmpReadId = (*sb_read_id)[i];

    // Check if Id is new -> initialize new read
    if (tmpReadId != currentReadId) {
      // New Headerline if new contigName. ContigName cannot change if ReadId
      // does not change
      std::string tmpCName = (*c_name)[i];
      if (tmpCName != currentCname) {
        // currentCname = string_columns[3][i]; DEBUG:
        // TODO uncomment if name is not with whitespaces and client is there
        //                                std::string headerLengthQuery =
        //                                "select count(rb_id) as length from "
        //                                        "reference_base join contig on
        //                                        rb_c_id=c_id where c_name=";
        //                                headerLengthQuery+=currentCname;
        //                                headerLengthQuery+=";";
        //                                TablePtr referenceBaseCount =
        //                                CoGaDB::SQL::executeSQL(headerLengthQuery,
        //                                client);
        //                                std::string currentHeaderLength =
        //                                referenceBaseCount->getColumnbyName("length")->getStringValue(0);
        headerOutStream << "@SQ\t"
                        << "SN:" << tmpCName << "\t"
                        << endl;  //"LN:" << currentHeaderLength << endl;
      }
      if (currentReadId != -1) {
        // FIXME put directly in samline calculation
        //                            currentCigarString = cigarFSM.getCigar();

        currentSamLine =
            currentQName + "\t" +
            boost::lexical_cast<std::string>(currentFlag) + "\t" +
            currentCname + "\t" +
            boost::lexical_cast<std::string>(currentSmallestRbPosition + 1) +
            "\t" + boost::lexical_cast<std::string>(currentMapQ) + "\t" +
            cigarFSM.getCigar() + "\t" + "*" + "\t" + "0" + "\t" + "0" + "\t" +
            currentBaseValueSequence + "\t" + currentQual;
        out << currentSamLine << endl;

        cigarFSM.reset();
        currentBaseValueSequence = "";
        currentQual = "";
      }

      currentReadId = tmpReadId;
      currentQName = (*r_qname)[i];
      currentFlag = (*r_flag)[i];
      currentCname = (*c_name)[i];

      currentSmallestRbPosition = (*rb_position)[i];
      currentMapQ = (*r_mapq)[i];
    }
    // Normal iteration skipps read-initialisation above

    std::string tmpBaseValue = (*sb_base_value)[i];
    cigarFSM.handleDatabaseInput(tmpBaseValue, (*sb_insert_offset)[i]);
    if (tmpBaseValue != "X") {
      currentBaseValueSequence += tmpBaseValue;
      char tmpMapQ = ((*sb_base_call_quality)[i] + 33);

      //                        int tmp = (*sb_base_call_quality)[i];
      //                        tmp+=33;
      //                        tmpMapQ = tmp;
      //
      currentQual += tmpMapQ;
    }
  }
  // Write last line FIXME - Better way please.. put one output at the end of
  // the loop
  out << currentQName + "\t" + boost::lexical_cast<std::string>(currentFlag) +
             "\t" + currentCname + "\t" +
             boost::lexical_cast<std::string>(currentSmallestRbPosition + 1) +
             "\t" + boost::lexical_cast<std::string>(currentMapQ) + "\t" +
             cigarFSM.getCigar() + "\t" + "*" + "\t" + "0" + "\t" + "0" + "\t" +
             currentBaseValueSequence + "\t" + currentQual;
  std::string result = headerStringStream.str() + ss.str();
  return result;
}
}
