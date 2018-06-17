/*
 * File:   samVerification.cpp
 * Author: john
 *
 * Created on 12. Juli 2015, 10:36
 */

#include "util/genomics_extension/samVerification.hpp"
#include <fstream>
#include <sstream>
#include <vector>
//#include <bits/stl_bvector.h>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>

namespace CoGaDB {

bool compareSamFiles(std::string originalSam, std::string exportedSam,
                     std::ostream& clientOut) {
  using std::endl;

  // Load in Samlines from Original Samfile
  std::ifstream original(originalSam.c_str());
  if (!original.good()) {
    clientOut << "Error: Could not open file: '" << originalSam << "'!" << endl;
    return false;
  }
  std::string line;
  std::vector<Samline> samlinesOriginalSam;
  bool missMatchFound = false;

  while (std::getline(original, line)) {
    // Skip Header
    if (line.at(0) == '@') {
      continue;
    }
    Samline tmpSamline;
    boost::split(tmpSamline, line, boost::is_any_of("\t"));
    unsigned int flag;
    try {
      flag = boost::lexical_cast<unsigned int>(tmpSamline[1]);
    } catch (const boost::bad_lexical_cast&) {
      clientOut << "Error: Line '";
      for (unsigned int i = 0; i < tmpSamline.size(); i++) {
        clientOut << tmpSamline[i];
      }
      clientOut << "' does not match Sam-Format!" << endl;
      return false;
    }
    // if 0x4 bit is set or cigar = *, the line is unmapped and therefore not in
    // database
    if (!(_isFlag0x4Set(flag) || (tmpSamline[5] == "*"))) {
      // cut out parts that are not supported by CoGaDB
      if (!_convertOriginalCigarForValidation(tmpSamline))
        return false;
      else {
        samlinesOriginalSam.push_back(tmpSamline);
      }
    }
  }

  // Load in Samlines from Exported sam and compare
  std::ifstream exported(exportedSam.c_str());
  if (!exported.good()) {
    clientOut << "Error: Could not open file: '" << exportedSam << "!" << endl;
    return false;
  }

  // clientOut << "DEBUG: Original-Sam has " << samlinesOriginalSam.size() << "
  // Lines at beginning!" << endl;

  while (std::getline(exported, line)) {
    // skip Header
    if (line.at(0) == '@') {
      continue;
    }

    Samline tmpSamline;
    boost::split(tmpSamline, line, boost::is_any_of("\t"));
    bool matchFound = false;
    for (unsigned int i = 0; i < samlinesOriginalSam.size(); i++) {
      if (_isSamlinesEqual(samlinesOriginalSam[i], tmpSamline)) {
        matchFound = true;
        samlinesOriginalSam.erase(samlinesOriginalSam.begin() + i);
        break;
      }
    }
    if (!matchFound) {
      missMatchFound = true;
      clientOut << "No match found for line ";
      for (unsigned int j = 0; j < tmpSamline.size(); j++) {
        clientOut << tmpSamline[j] << " ";
      }
      clientOut << " from exported Samfile!" << endl;
    }
  }

  if (missMatchFound) {
    clientOut << endl
              << endl
              << "_____________________________________________________________"
                 "_________________________________"
              << endl;
  }
  if (samlinesOriginalSam.size() > 0) {
    //            clientOut << samlinesOriginalSam.size() << " lines from
    //            original Samfile are not exported." << endl;
    clientOut << "_____________________________________________________________"
                 "_________________________________"
              << endl
              << endl;
    for (unsigned int i = 0; i < samlinesOriginalSam.size(); i++) {
      clientOut << "Error: Line ";
      for (unsigned int j = 0; j < samlinesOriginalSam[i].size(); j++) {
        clientOut << samlinesOriginalSam[i].at(j) << " ";
      }
      clientOut << " from original Samfile does not get exportet!" << endl;
    }
  }
  if (missMatchFound || samlinesOriginalSam.size() > 0) {
    clientOut
        << "====> Original-Sam and Exported-Sam are functionally different."
        << endl;
    return false;
  } else {
    clientOut << "====> Original-Sam and Exported-Sam are functionally equal."
              << endl;
    return true;
  }
}

bool _isSamlinesEqual(const Samline& originalSamline,
                      const Samline& exportedSamline) {
  /* Following columns are checked for equality
   * 0 = QNAME
   * 1 = FLAG
   * 2 = RNAME
   * 3 = POS
   * 4 = MAPQ
   * 5 = CIGAR
   * 9 = SEQ
   * 10 = QUAL
   */
  return ((originalSamline[0] == exportedSamline[0]) &&
          (originalSamline[1] == exportedSamline[1]) &&
          (originalSamline[2] == exportedSamline[2]) &&
          (originalSamline[3] == exportedSamline[3]) &&
          (originalSamline[4] == exportedSamline[4]) &&
          (originalSamline[5] == exportedSamline[5]) &&
          (originalSamline[9] == exportedSamline[9]) &&
          (originalSamline[10] == exportedSamline[10]));
}

bool _isFlag0x4Set(int flag) { return ((flag & 4) == 4); }

bool _convertOriginalCigarForValidation(Samline& samline) {
  std::string cigar(samline[5]);
  std::string seq(samline[9]);
  std::string qual(samline[10]);

  boost::regex numberReg("\\d");

  int positionCounterSeq(0);
  int positionCounterCigar(0);
  std::string currentSub("");

  for (unsigned int i = 0; i < cigar.size(); i++) {
    std::string currentChar("");
    currentChar += cigar[i];
    if (boost::regex_match(currentChar, numberReg)) {
      currentSub += cigar[i];
    } else {
      int preNumber;
      try {
        preNumber = boost::lexical_cast<int>(currentSub);
      } catch (const boost::bad_lexical_cast&) {
        return false;
      }
      switch (cigar[i]) {
        case 'M':
        case 'I':
        case 'X':
        case '=':
          // M,I,X,= Move only position Pointer
          positionCounterSeq += preNumber;
          break;
        case 'S':
          // S cut from Cigar, Qual and Seq
          cigar.erase(positionCounterCigar, i - positionCounterCigar + 1);
          seq.erase(positionCounterSeq, preNumber);
          qual.erase(positionCounterSeq, preNumber);
          i = i - currentSub.size() - 1;
          break;
        case 'H':
        case 'P':
        case 'N':
          // H,P,N cut from Cigar-String
          cigar.erase(positionCounterCigar, i - positionCounterCigar + 1);
          i = i - currentSub.size() - 1;
          break;
        case 'D':
          // Do nothing. Keep in Cigar, SEQ and Qual but don't move position
          // Counter
          break;
        default:
          return false;
      }
      currentSub = "";
      positionCounterCigar = i + 1;
    }
  }
  samline[5] = cigar;
  samline[9] = seq;
  samline[10] = qual;
  return true;
}

}  // end namespace CogaDB
