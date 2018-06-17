/*
 * File:   samExportCommandParser.hpp
 * Author: john
 *
 * Created on 4. Juli 2015, 19:26
 */

#pragma once

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "util/genomics_extension/samQueryCreator.hpp"  //can get removed if namespace cogadb?

namespace CoGaDB {
  // If Filename is not Declared, it will
  const std::string SAM_EXPORTER_SIMPLE_INPUT_API_DEFINITION =
      "<contigID_1>,<contigID_2>,...,<contigID_n> <start> <end> -<param_1> ... "
      "-<param_n> (<Filename>)";
  const std::string DEFAULT_SAM_EXPORT_FILENAME = "default";

  class SamExportParser {
   public:
    SamExportParser();
    SamExportParser(const SamExportParser& orig);
    ~SamExportParser();
    bool isTimeKeepingEnabled();
    bool isIgvControlEnabled();
    bool isSnapshotEnabled();
    SamQueryCreator::SamIdSelectionPtr getSamSelection();
    std::string getSavePath();
    virtual bool addArguments(std::vector<std::string> newArg,
                              std::ostream& clientOut) = 0;
    int getStartValue();
    int getEndValue();

   protected:
    typedef bool (
        SamExportParser::*SamExportParameterHandlerPtr)();  // Member Function
                                                            // Pointer takes no
                                                            // arguments and
                                                            // returns a bool
    typedef std::map<std::string, SamExportParameterHandlerPtr>
        SamExportParameter;
    typedef std::map<std::string, SamExportParameterHandlerPtr>::iterator
        SamExportParameterIterator;
    SamExportParameter samExportParameter;
    SamExportParameterIterator samExportIterator;
    SamQueryCreator::SamIdSelectionPtr samSelection;
    bool enableTimeKeeping();
    bool enableIgvControl();
    bool enableSnapshot();
    void setNewSaveName(const std::string newSaveName);
    int startValue;
    int endValue;

   private:
    bool timeKeepingEnabled;
    bool igvControlEnabled;
    bool snapshotEnabled;
    std::string savePath;
  };

  class SimpleSamExportParser : public SamExportParser {
   public:
    SimpleSamExportParser();
    // SimpleSamExportParser(const SimpleSamExportParser& orig);
    ~SimpleSamExportParser();
    bool addArguments(std::vector<std::string> newArg, std::ostream& clientOut);

   private:
    bool isContigWithStar(const std::vector<std::string>& contigs);
  };

}  // end namespace CogaDB