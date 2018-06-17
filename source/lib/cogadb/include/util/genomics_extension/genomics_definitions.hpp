//
// Created by Sebastian Dorok on 06.07.15.
//

#include <boost/chrono.hpp>
#include <iomanip>
#include <util/time_measurement.hpp>

#include <cmath>

#ifndef GPUDBMS_GENOMICS_DEFINITIONS_HPP
#define GPUDBMS_GENOMICS_DEFINITIONS_HPP

namespace CoGaDB {

  using namespace boost::chrono;

  const std::string GENOME_SCHEMA_TYPE_PARAMETER = "genome_schema_type";
  const std::string GENOME_IMPORTER_VERBOSE_PARAMETER =
      "genome_importer_verbose";
  const std::string BASE_CENTRIC_SCHEMA_TYPE_PARAMETER_VALUE = "base_centric";
  const std::string STORAGE_EXPERIMENTS_SCHEMA_TYPE_PARAMETER_VALUE =
      "storage_experiments";
  const std::string SEQUENCE_CENTRIC_SCHEMA_TYPE_PARAMETER_VALUE =
      "sequence_centric";
  const std::string SEQUENCE_CENTRIC_SCHEMA_WITH_STASH_TYPE_PARAMETER_VALUE =
      "sequence_centric_with_stashing";

  const std::string GENOME_SCHEMA_COMPRESSION_PARAMETER =
      "genome_schema_compression";

  const std::string GENOTYPE_FREQUENCY_MIN_PARAMETER = "genotype_frequency_min";
  const std::string GENOTYPE_FREQUENCY_MAX_PARAMETER = "genotype_frequency_max";

  // Helper methods

  /*  */
  inline void _printProgress(uint64_t cur, float prog, uint32_t max,
                             std::ostream &out, bool final) {
    out << std::fixed << std::setprecision(2) << "\r   ["
        << std::string(static_cast<std::string::size_type>(cur), '#')
        << std::string(max - cur, ' ') << "] " << 100 * prog << "%";

    if (prog == 1.f && final) {
      out << std::endl;
    } else {
      out.flush();
    }
  }

  /* Prints a progress bar
   *
   * Params:
   * lengthOverall    .. maximum possible value
   * length           .. current value
   * maximumIndicatos .. maximum number of indicator signs '#'
   */
  inline void _drawProgress(uint64_t maximumValue, uint64_t currentValue,
                            uint32_t maximumIndicator, std::ostream &out,
                            bool final = false) {
    float progress(
        currentValue /
        static_cast<float>(maximumValue));  // percentage of infile already read

    // Number of #'s as function of current progress
    uint64_t cur(static_cast<uint64_t>(std::ceil(progress * maximumIndicator)));
    _printProgress(cur, progress, maximumIndicator, out, final);
  }

  /**
   * Returns 0 if suffix is the suffix of string.
   */
  inline int _endsWithFoo(const char *string, const char *suffix) {
    string = strrchr(string, '.');

    if (string != NULL && suffix != NULL) return (strcmp(string, suffix));

    return (-1);
  }
}

#endif  // GPUDBMS_GENOMICS_DEFINITIONS_HPP
