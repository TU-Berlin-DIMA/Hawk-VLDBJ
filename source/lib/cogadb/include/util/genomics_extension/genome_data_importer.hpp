//
// Created by Sebastian Dorok on 06.07.15.
//

//#pragma once

#ifndef GENOME_DATA_IMPORTER_FILES_HPP
#define GENOME_DATA_IMPORTER_FILES_HPP

#ifdef BAM_FOUND

#include <samtools/bam.h>
#include <samtools/sam.h>
#include <core/compressed_column.hpp>
#include <core/global_definitions.hpp>
#include <util/genomics_extension/alignment_files.hpp>
#include <util/genomics_extension/genomics_definitions.hpp>

namespace CoGaDB {
  using namespace std;

  class GenomeDataImporter {
   protected:
    typedef ColumnBaseTyped<int> IntTypedColumnType;
    typedef boost::shared_ptr<IntTypedColumnType> IntTypedColumnPtr;
    typedef ColumnBaseTyped<TID> OIDTypedColumnType;
    typedef boost::shared_ptr<OIDTypedColumnType> OIDTypedColumnPtr;
    typedef ColumnBaseTyped<string> StringTypedColumnType;
    typedef boost::shared_ptr<StringTypedColumnType> StringTypedColumnPtr;

    TablePtr refGenomeTable;
    TablePtr contigTable;
    TablePtr refBaseTable;
    TablePtr sampleGenomeTable;
    TablePtr readTable;
    TablePtr unmappedReadsTable;
    TablePtr readAuxTable;
    TablePtr unmappedReadsAuxTable;
    TablePtr sampleBaseTable;

    struct contig_ids {
      TID id_of_contig;
      TID id_of_first_reference_base;

      contig_ids(TID _id_of_contig, TID _id_of_first_reference_base)
          : id_of_contig(_id_of_contig),
            id_of_first_reference_base(_id_of_first_reference_base){};
    };

    struct aux_info {
      string tag;
      string type;
      string value;

      aux_info(string _tag, string _type, string _value)
          : tag(_tag), type(_type), value(_value){};
    };

    struct cigar_conversion {
      std::vector<uint32_t> positions;
      /* array to store insert_offsets */
      std::vector<uint32_t> insert_offsets;
      /* array to store positions for deletions */
      /* we assume that there at most the number of available read bases as
       * deletions */
      std::vector<uint32_t> deleted_positions;
      /* soft clip offsets */
      int start_offset;
      int end_offset;
      int del_end_offset;

      cigar_conversion(int32_t size) {
        positions = std::vector<uint32_t>(size);
        insert_offsets = std::vector<uint32_t>(size);
        deleted_positions = std::vector<uint32_t>(size);
        start_offset = 0;
        end_offset = 0;
        del_end_offset = 0;
      };
    };

    void _convertCIGAR(uint32_t *cigar, uint32_t cigarLength, int startPos,
                       cigar_conversion *converted_cigar);

    ostream &out;

    bool verbose;

    bool compress_data;

    std::pair<double, std::string> _getTimeAsHumanReadableString(
        Timestamp time_in_nanoseconds);

   public:
    /**
    * \brief Constructor.
    *
    * @param client Client enables access to system resources such as output
    *streams.
    */
    GenomeDataImporter(ostream &out, bool compress_data, bool verbose);

    /**
    * \brief Default destructor.
    */
    virtual ~GenomeDataImporter();

    /**
    * \brief Method imports genome data from a FASTA file using the appropriate
    *database schema into the database.
    *
    * @param path_to_fasta_file Path to FASTA file for import.
    * @param reference_genome_name Name of the imported reference genome. If
    *left out or empty, the file name without
    * file ending is used as name.
    */
    virtual bool importReferenceGenomeData(
        const string path_to_fasta_file,
        const string reference_genome_name = "") = 0;

    /**
    * \brief Method imports genome data from a SAM or BAM file using the
    *appropriate database schema into the database.
    *
    * @param path_to_sambam_file Path to SAM or BAM file for import.
    * @param reference_genome_name Name of the reference genome used for
    *aligning the sample genome.
    * @param sample_genome_name Name of the imported sample genome. If left out
    *or empty, the file name without file
    * ending is used as name.
    */
    virtual bool importSampleGenomeData(
        const string path_to_sambam_file, const string reference_genome_name,
        const string sample_genome_name = "") = 0;
  };

}  // end namespace CoGaDB

#endif

#endif  // GENOME_DATA_IMPORTER_FILES_HPP
