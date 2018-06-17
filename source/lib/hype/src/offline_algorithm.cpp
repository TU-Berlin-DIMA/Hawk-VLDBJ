/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#include <core/offline_algorithm.hpp>
#include <core/operation.hpp>
#include <core/plotscriptgenerator.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>

#include <core/scheduler.hpp>

namespace hype {
namespace core {

const std::string& Offline_Algorithm::getAlgorithmName() const throw() {
  return name_;
}

const std::string& Offline_Algorithm::getOperationName() const throw() {
  return operation_name_;
}

DeviceSpecification Offline_Algorithm::getDeviceSpecification() const throw() {
  return device_;
}

unsigned int Offline_Algorithm::getNumberOfMeasurementPairs() const throw() {
  return offline_mesurement_pairs_.size();
}

Offline_Algorithm::Offline_Algorithm(DeviceSpecification device,
                                     std::string algorithm_name,
                                     std::string opname, std::string filepath)
    : offline_mesurement_pairs_(),
      device_(device),
      name_(algorithm_name),
      operation_name_(opname),
      current_mesurementpair_index_(0),
      filepath_(filepath) {
  if (!quiet)
    std::cout << "Loading data from file: '" << filepath << "'..." << std::endl;
  loadMeasurementpairsfromFile(filepath);

  if (!quiet && verbose && debug) printstoredMeasurementpairs();
}

core::MeasurementPair Offline_Algorithm::getNext() {
  if (offline_mesurement_pairs_.size() > current_mesurementpair_index_)
    return offline_mesurement_pairs_[current_mesurementpair_index_++];
  else
    return core::MeasurementPair();  // null sample, indicator that all samples
                                     // were processed
}

bool Offline_Algorithm::hasNext() {
  return (offline_mesurement_pairs_.size() > current_mesurementpair_index_);
}

void Offline_Algorithm::reset() { current_mesurementpair_index_ = 0; }

// the first line of a file determines, how many feature values are expected:
// for n values, the first n-1 values are considered feature Values and the
// remaining 1 value is considered the measured execution time
void Offline_Algorithm::loadMeasurementpairsfromFile(std::string filepath) {
  std::ifstream fin(filepath.c_str());  //"Makefile");
  std::string buffer;

  if (!quiet && verbose && debug) std::cout << "Hier der Inhalt der Datei:\n";

  unsigned int samplecounter = 0;
  unsigned int number_of_feature_values = 0;

  while (fin.good()) {
    getline(fin, buffer, '\n');
    if (buffer == "") break;

    if (!quiet && verbose && debug)
      std::cout << samplecounter << " " << buffer << std::endl;

    // std::string s = buffer;
    std::string str = ";;Hello|world||-foo--bar;yow;baz|";
    str = buffer;  // s;

    // std::string xvalue="";
    // std::string yvalue="";

    typedef boost::tokenizer<boost::char_separator<char> > tokenizer;
    boost::char_separator<char> sep("\t");
    tokenizer tokens(str, sep);

    if (samplecounter == 0) {
      for (tokenizer::iterator tok_iter = tokens.begin();
           tok_iter != tokens.end(); ++tok_iter)
        number_of_feature_values++;

      number_of_feature_values--;  // last value is for measured value
    }

    unsigned int tokencounter = 0;
    Tuple t;
    MeasuredTime time;
    for (tokenizer::iterator tok_iter = tokens.begin();
         tok_iter != tokens.end(); ++tok_iter) {
      if (tokencounter < number_of_feature_values) {
        double d = boost::lexical_cast<double>(*tok_iter);
        t.push_back(d);
      } else {
        time = MeasuredTime(boost::lexical_cast<double>(*tok_iter));
      }
      tokencounter++;
    }
    assert(number_of_feature_values == t.size());
    offline_mesurement_pairs_.push_back(
        core::MeasurementPair(t, time, EstimatedTime(0)));
    samplecounter++;
  }
  // std::cout << "\nEnde der Ausgabe\n";
  fin.close();
}

void Offline_Algorithm::storeInFile(const std::string& file_name) {
  std::fstream file(file_name.c_str(),
                    std::ios_base::out | std::ios_base::trunc);
  for (unsigned int i = 0; i < offline_mesurement_pairs_.size(); i++) {
    file << offline_mesurement_pairs_[i].toPlainString()
         << std::endl;  //.datasize << "\t" <<
    // offline_mesurement_pairs_[i].measured_execution_time
    //<< std::endl;
    // std::cout << offline_mesurement_pairs_[i].datasize << " ";
  }
  // std::cout << std::endl;
}

void Offline_Algorithm::printstoredMeasurementpairs() {
  for (unsigned int i = 0; i < offline_mesurement_pairs_.size(); i++) {
    std::cout << offline_mesurement_pairs_[i]
              << std::endl;  //.datasize << "\t" <<
    // offline_mesurement_pairs_[i].measured_execution_time <<
    // std::endl;
    // std::cout << offline_mesurement_pairs_[i].datasize << " ";
  }
  // std::cout << std::endl;
}

std::vector<Offline_Algorithm>
Offline_Algorithm::randomize_dataset_of_offline_algorithms(
    std::vector<Offline_Algorithm> offline_algorithms) {
  std::vector<Offline_Algorithm> tmp_algorithms(
      offline_algorithms);  // copy vector
  std::vector<Offline_Algorithm> result_algorithms(offline_algorithms);
  // std::vector<Offline_Algorithm>
  // result_algorithms(offline_algorithms.size()); //create vector as big as
  // offline_algorithms (objects are default constructed)

  for (unsigned int j = 0; j < result_algorithms.size(); j++) {
    result_algorithms[j].offline_mesurement_pairs_.clear();  // delete samples,
                                                             // we want a clean
                                                             // vector to put
                                                             // the randomized
                                                             // sampels in
  }

  unsigned long long number_of_input_measurement_values =
      offline_algorithms[0].offline_mesurement_pairs_.size();

  for (unsigned int i = 0; i < number_of_input_measurement_values; i++) {
    unsigned int random_index =
        rand() % tmp_algorithms[0].offline_mesurement_pairs_.size();
    if (!quiet && verbose && debug)
      std::cout << "random index: " << random_index << std::endl;
    for (unsigned int j = 0; j < tmp_algorithms.size(); j++) {
      core::MeasurementPair pair =
          tmp_algorithms[j].offline_mesurement_pairs_[random_index];
      result_algorithms[j].offline_mesurement_pairs_.push_back(pair);
      tmp_algorithms[j].offline_mesurement_pairs_.erase(
          tmp_algorithms[j].offline_mesurement_pairs_.begin() +
          random_index);  // random_index
    }
  }

  return result_algorithms;
}

}  // end namespace core
}  // end namespace hype
