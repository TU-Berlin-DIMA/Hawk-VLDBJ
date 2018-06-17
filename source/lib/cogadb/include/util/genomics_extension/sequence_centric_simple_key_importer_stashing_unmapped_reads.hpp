//
// Created by Sebastian Dorok on 06.07.15.
//

//#pragma once

#ifndef SEQUENCE_CENTRIC_SIMPLE_KEY_WITH_STASH_IMPORTER_FILES_HPP
#define SEQUENCE_CENTRIC_SIMPLE_KEY_WITH_STASH_IMPORTER_FILES_HPP

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
#include <util/genomics_extension/sequence_centric_simple_key_importer.hpp>

namespace CoGaDB {
  using namespace std;

  // table and column names
  const string UR_TBL_NAME = "UREAD";
  const string UR_ID_COL_NAME = "UR_ID";
  const string UR_QNAME_COL_NAME = "UR_QNAME";
  const string UR_MAPPING_QUALITY_COL_NAME = "UR_MAPQ";
  const string UR_FLAG_COL_NAME = "R_FLAG";
  const string UR_SAMPLE_GENOME_ID_COL_NAME = "UR_SG_ID";
  const string UR_MATE_ID_COL_NAME = "UR_MATE_ID";
  const string UR_C_ID_COL_NAME = "UR_C_ID";
  const string UR_POS_COL_NAME = "UR_POS";
  const string UR_CIGAR_COL_NAME = "UR_CIGAR";
  const string UR_SEQ_COL_NAME = "UR_SEQ";
  const string UR_QUAL_COL_NAME = "UR_QUAL";

  const string URAUX_TBL_NAME = "UREAD_AUX";
  const string URAUX_ID_COL_NAME = "URAUX_ID";
  const string URAUX_READ_ID_COL_NAME = "URAUX_READ_ID";
  const string URAUX_TAG_COL_NAME = "URAUX_TAG";
  const string URAUX_TYPE_COL_NAME = "URAUX_TYPE";
  const string URAUX_VALUE_COL_NAME = "URAUX_VALUE";

  class SequenceCentric_SimpleKey_WithStash_Importer
      : public GenomeDataImporter {
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
    OIDTypedColumnPtr r_c_id_column;
    IntTypedColumnPtr r_pos_column;
    IntTypedColumnPtr r_map_qual_column;
    StringTypedColumnPtr r_cigar_column;
    StringTypedColumnPtr r_seq_column;
    StringTypedColumnPtr r_qual_column;
    OIDTypedColumnPtr r_sg_id_column;
    OIDTypedColumnPtr r_mate_id_column;

    OIDTypedColumnPtr ur_id_column;
    StringTypedColumnPtr ur_qname_column;
    IntTypedColumnPtr ur_flag_column;
    OIDTypedColumnPtr ur_c_id_column;
    IntTypedColumnPtr ur_pos_column;
    IntTypedColumnPtr ur_map_qual_column;
    StringTypedColumnPtr ur_cigar_column;
    StringTypedColumnPtr ur_seq_column;
    StringTypedColumnPtr ur_qual_column;
    OIDTypedColumnPtr ur_sg_id_column;
    OIDTypedColumnPtr ur_mate_id_column;

    OIDTypedColumnPtr read_aux_id_column;
    OIDTypedColumnPtr read_aux_read_id_column;
    StringTypedColumnPtr read_aux_tag_column;
    StringTypedColumnPtr read_aux_type_column;
    StringTypedColumnPtr read_aux_value_column;

    OIDTypedColumnPtr uread_aux_id_column;
    OIDTypedColumnPtr uread_aux_read_id_column;
    StringTypedColumnPtr uread_aux_tag_column;
    StringTypedColumnPtr uread_aux_type_column;
    StringTypedColumnPtr uread_aux_value_column;

    bool _createDatabase();

    bool _importReferenceGenomeData(const string reference_genome_name,
                                    const string path_to_fasta_file);

    bool _importAlignment(TID sample_genome_id, TID id_of_contig,
                          bam1_t *alignment, const bam1_core_t c, TID read_id,
                          TID mate_id, TID &raux_id, string converted_cigar);

    bool _importUnalignedAlignment(TID sample_genome_id, TID id_of_contig,
                                   bam1_t *alignment, const bam1_core_t c,
                                   TID read_id, TID mate_id, TID &raux_id,
                                   string converted_cigar);

    bool _importSampleGenomeData(string sample_genome_name,
                                 TID reference_genome_id,
                                 string path_to_sam_bam_file,
                                 TID &numberOfUnmappedReads,
                                 TID &numberOfReads);

    /**
     * Initiates connection to database by initializing column pointers. If
     * database does not exist, this method creates it.
     */
    bool _initDatabaseConnection();

    std::vector<GenomeDataImporter::aux_info> getAuxiliaryStringVector(
        bam1_t *alignment);

    string convertCIGAR2String(uint32_t *cigar, uint32_t cigarLength);

   public:
    SequenceCentric_SimpleKey_WithStash_Importer(ostream &,
                                                 bool compress_data = true,
                                                 bool verbose = true);

    bool importReferenceGenomeData(const string path_to_fasta_file,
                                   const string reference_genome_name);

    bool importSampleGenomeData(const string path_to_sambam_file,
                                const string reference_genome_name,
                                const string sample_genome_name);
  };

}  // end namespace CoGaDB

#endif  // BAM_FOUND

#endif  // SEQUENCE_CENTRIC_SIMPLE_KEY_WITH_STASH_IMPORTER_FILES_HPP
