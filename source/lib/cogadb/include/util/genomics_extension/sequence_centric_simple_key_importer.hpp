//
// Created by Sebastian Dorok on 06.07.15.
//

//#pragma once

#ifndef SEQUENCE_CENTRIC_SIMPLE_KEY_IMPORTER_FILES_HPP
#define SEQUENCE_CENTRIC_SIMPLE_KEY_IMPORTER_FILES_HPP

#ifdef BAM_FOUND

#include <compression/bitpacked_dictionary_compressed_column.hpp>
#include <compression/bitpacked_dictionary_compressed_column_base_values.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <compression/dictionary_compressed_column_base_values.hpp>
#include <compression/rle_compressed_column.hpp>
#include <compression/rle_delta_one_compressed_column_int.hpp>
#include <compression/void_compressed_column_int.hpp>
#include <core/table.hpp>
#include <parser/client.hpp>
#include <query_processing/query_processor.hpp>
#include <util/genomics_extension/genome_data_importer.hpp>

#include <util/genomics_extension/base_centric_simple_key_importer.hpp>

namespace CoGaDB {
  using namespace std;

  // table and column names
  const string C_SEQ_COL_NAME = "C_SEQ";

  const string R_RNAME_COL_NAME = "R_RNAME";
  const string R_POS_COL_NAME = "R_POS";
  const string R_CIGAR_COL_NAME = "R_CIGAR";
  const string R_SEQ_COL_NAME = "R_SEQ";
  const string R_QUAL_COL_NAME = "R_QUAL";
  const string R_C_ID_COL_NAME = "R_C_ID";

  const string RAUX_TBL_NAME = "READ_AUX";
  const string RAUX_ID_COL_NAME = "RAUX_ID";
  const string RAUX_READ_ID_COL_NAME = "RAUX_READ_ID";
  const string RAUX_TAG_COL_NAME = "RAUX_TAG";
  const string RAUX_TYPE_COL_NAME = "RAUX_TYPE";
  const string RAUX_VALUE_COL_NAME = "RAUX_VALUE";

  class SequenceCentric_SimpleKey_Importer : public GenomeDataImporter {
   private:
    OIDTypedColumnPtr rg_id_column;
    StringTypedColumnPtr rg_name_column;

    OIDTypedColumnPtr c_id_column;
    StringTypedColumnPtr c_name_column;
    StringTypedColumnPtr c_seq_column;
    OIDTypedColumnPtr c_rg_id_column;

    OIDTypedColumnPtr sg_id_column;
    StringTypedColumnPtr sg_name_column;

    OIDTypedColumnPtr r_id_column;
    StringTypedColumnPtr r_qname_column;
    IntTypedColumnPtr r_flag_column;
    StringTypedColumnPtr r_rname_column;
    IntTypedColumnPtr r_pos_column;
    IntTypedColumnPtr r_map_qual_column;
    StringTypedColumnPtr r_cigar_column;
    StringTypedColumnPtr r_seq_column;
    StringTypedColumnPtr r_qual_column;
    OIDTypedColumnPtr r_sg_id_column;
    OIDTypedColumnPtr r_c_id_column;
    OIDTypedColumnPtr r_mate_id_column;

    OIDTypedColumnPtr read_aux_id_column;
    OIDTypedColumnPtr read_aux_read_id_column;
    StringTypedColumnPtr read_aux_tag_column;
    StringTypedColumnPtr read_aux_type_column;
    StringTypedColumnPtr read_aux_value_column;

    bool _createDatabase();

    bool _importReferenceGenomeData(const string reference_genome_name,
                                    const string path_to_fasta_file);

    bool _importAlignment(TID sample_genome_id, TID id_of_contig,
                          bam1_t *alignment, const bam1_core_t c, TID read_id,
                          TID mate_id, TID &raux_id, string converted_cigar);

    bool _importSampleGenomeData(string sample_genome_name,
                                 TID reference_genome_id,
                                 string path_to_sam_bam_file,
                                 TID &numberOfUnmappedReads,
                                 TID &numberOfReads);

    std::vector<GenomeDataImporter::aux_info> getAuxiliaryStringVector(
        bam1_t *alignment);

    string convertCIGAR2String(uint32_t *cigar, uint32_t cigarLength);

    /**
     * Initiates connection to database by initializing column pointers. If
     * database does not exist, this method creates it.
     */
    bool _initDatabaseConnection();

   public:
    SequenceCentric_SimpleKey_Importer(ostream &, bool compress_data = true,
                                       bool verbose = true);

    bool importReferenceGenomeData(const string path_to_fasta_file,
                                   const string reference_genome_name);

    bool importSampleGenomeData(const string path_to_sambam_file,
                                const string reference_genome_name,
                                const string sample_genome_name);
  };

}  // end namespace CoGaDB

#endif  // BAM_FOUND

#endif  // SEQUENCE_CENTRIC_SIMPLE_KEY_IMPORTER_FILES_HPP
