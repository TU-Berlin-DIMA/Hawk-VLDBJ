/*
 * File:   samExportCommandParser.cpp
 * Author: john
 *
 * Created on 4. Juli 2015, 19:26
 */
#include "util/genomics_extension/samExportCommandParser.hpp"
#include <boost/algorithm/string.hpp>
#include "core/variable_manager.hpp"
#include "parser/client.hpp"

namespace CoGaDB {
using namespace std;

SamExportParser::SamExportParser() {
  // Define Parameter for Command
  samExportParameter.insert(
      std::make_pair("-t", &SamExportParser::enableTimeKeeping));
  samExportParameter.insert(
      std::make_pair("-i", &SamExportParser::enableIgvControl));
  samExportParameter.insert(
      std::make_pair("-s", &SamExportParser::enableSnapshot));

  // Set Member-Variables to default value
  timeKeepingEnabled = false;
  igvControlEnabled = false;
  startValue = 0;
  endValue = 0;

  savePath =
      VariableManager::instance().getVariableValueString("sam_save_path") +
      "/" + DEFAULT_SAM_EXPORT_FILENAME + ".sam";

  // Initialisation
  samSelection =
      SamQueryCreator::SamIdSelectionPtr(new SamQueryCreator::SamIdSelection());
}

bool SamExportParser::enableTimeKeeping() {
  timeKeepingEnabled = true;
  return true;
}

bool SamExportParser::isTimeKeepingEnabled() { return timeKeepingEnabled; }

bool SamExportParser::enableIgvControl() {
  igvControlEnabled = true;
  return true;
}

bool SamExportParser::isIgvControlEnabled() { return igvControlEnabled; }

bool SamExportParser::enableSnapshot() {
  snapshotEnabled = true;
  return true;
}

bool SamExportParser::isSnapshotEnabled() { return snapshotEnabled; }

SamQueryCreator::SamIdSelectionPtr SamExportParser::getSamSelection() {
  return samSelection;
}

SamExportParser::~SamExportParser() {}

void SamExportParser::setNewSaveName(const std::string newSaveName) {
  savePath =
      VariableManager::instance().getVariableValueString("sam_save_path");
  savePath += "/";
  savePath += newSaveName;
  savePath += ".sam";
}

std::string SamExportParser::getSavePath() { return savePath; }

int SamExportParser::getStartValue() {
  return startValue;  // Database starts position with 0 so we add 1 to match
                      // normal position
}

int SamExportParser::getEndValue() {
  return endValue;  // Database starts position with 0 so we add 1 to match
                    // normal position
}

/**
 * Parser for Simple Input Api.
 */

SimpleSamExportParser::SimpleSamExportParser() {
  // do nothing, just call constructor above...
}

bool SimpleSamExportParser::addArguments(std::vector<std::string> newArg,
                                         std::ostream& clientOut) {
  // SamExportParser();
  std::vector<std::string> contigIdsForSamQueryBuilder;
  boost::split(contigIdsForSamQueryBuilder, newArg[0], boost::is_any_of(","));

  // checking for valid number of arguments
  if (newArg.size() < 3) {
    clientOut << "Error: Not enough Arguments." << endl;
    return false;
  }

  // handling * in input
  // Start is *
  if (newArg[1] == "*") {
    newArg[1] = "-1";
  }
  // End is *
  if (newArg[2] == "*") {
    newArg[2] = "-1";
  }

  // Check for valid range
  bool startTransform = false;
  bool endTransform = false;
  try {
    startValue = boost::lexical_cast<int>(newArg[1]);
    endValue = boost::lexical_cast<int>(newArg[2]);
    // database Position starts at 0 while SAM file position starts at 1
    // FIXME make clean code
    if (startValue > 0) {
      startValue--;
      startTransform = true;
    }
    if (endValue > 0) {
      endValue--;
      endTransform = true;
    }

  } catch (const boost::bad_lexical_cast&) {
    clientOut << "Error: Values for Start or End cannot be cast to Integer!"
              << endl;
    return false;
  }

  if (startValue < -1) {
    clientOut << "Error: Start has to be greater than or equal to 0." << endl;
    return false;
  }
  if (endValue < -1) {
    clientOut << "Error: End has to be greater than or equal to 0." << endl;
    return false;
  }
  if (endValue != -1 && endValue < startValue) {
    clientOut << "Error: End has to be greater than or equal to Start." << endl;
    return false;
  }
  // If there is a star in one of the contigIds, it consumes all others
  if (isContigWithStar(contigIdsForSamQueryBuilder)) {
    samSelection->push_back(boost::make_tuple(startValue, endValue, -1));
  } else {
    // Parse in contigIDs one by one
    for (unsigned int i = 0; i < contigIdsForSamQueryBuilder.size(); i++) {
      try {
        int currentContigId =
            boost::lexical_cast<int>(contigIdsForSamQueryBuilder[i]);
        // ContigID has to be greater than or equal to 0
        if (currentContigId < 0) {
          clientOut << "Error: " << i + 1
                    << ". ContigID is not greater than or equal to 0." << endl;
          return false;
        }
        samSelection->push_back(
            boost::make_tuple(startValue, endValue, currentContigId));
      } catch (const boost::bad_lexical_cast&) {
        clientOut << "Error: " << i + 1
                  << ". ContigID cannot be cast to Integer." << endl;
        return false;
      }
    }
  }
  // Parse Parameter
  for (unsigned i = 3; i < newArg.size(); i++) {
    // If last Parameter has no -, take as Filename
    if (!(newArg[i].at(0) == '-') && i == newArg.size() - 1) {
      setNewSaveName(newArg[i]);
      break;
    }
    //
    //        std::vector<std::string> parameterSplit;
    //        boost::split(parameterSplit, newArg[i], boost::is_any_of("="));

    samExportIterator = samExportParameter.find(newArg[i]);
    if (samExportIterator != samExportParameter.end()) {
      SamExportParameterHandlerPtr handlerFunction = samExportIterator->second;
      if (!(this->*handlerFunction)()) {
        clientOut << "Error: Executing Parameter " << newArg[i] << " failed!"
                  << endl;
        return false;
      }
    } else {
      clientOut << "Error: Parameter " << newArg[i] << " is not defined!"
                << endl;
      return false;
    }
  }
  // indextransformation back. FIXME make clean code
  if (startTransform) startValue++;
  if (endTransform) endValue++;
  return true;
}

bool SimpleSamExportParser::isContigWithStar(
    const std::vector<std::string>& contigs) {
  for (unsigned int i = 0; i < contigs.size(); i++) {
    if (contigs[i] == "*") return true;
  }
  return false;
}

SimpleSamExportParser::~SimpleSamExportParser() {
  // do nothing... just call constructor above
}

}  // end namespace CogaDB
