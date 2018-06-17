/*
 * File:   samVerification.hpp
 * Author: john
 *
 * Created on 12. Juli 2015, 10:36
 */

#pragma once

#include <string>
#include <vector>

namespace CoGaDB {
  typedef std::vector<std::string> Samline;

  bool compareSamFiles(std::string originalSam, std::string exportedSam,
                       std::ostream& clientOut);
  bool _isFlag0x4Set(int flag);
  bool _convertOriginalCigarForValidation(Samline& samline);
  bool _isSamlinesEqual(const Samline& originalSamline,
                        const Samline& exportedSamline);
}  // end namespace CogaDB