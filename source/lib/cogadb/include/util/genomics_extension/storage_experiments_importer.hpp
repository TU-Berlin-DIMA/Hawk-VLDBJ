//
// Created by Sebastian Dorok on 06.07.15.
//

//#pragma once
#ifndef STORAGE_EXPERIMENTS_IMPORTER_FILES_HPP
#define STORAGE_EXPERIMENTS_IMPORTER_FILES_HPP

#ifdef BAM_FOUND

#include <compression/bitpacked_dictionary_compressed_column.hpp>
#include <compression/bitpacked_dictionary_compressed_column_base_values.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <compression/dictionary_compressed_column_base_values.hpp>
#include <compression/reference_based_compressed_column.hpp>
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

  // additonal table and column names

  const string RG_ID_VOID_COL_NAME = "RG_ID_void";

  const string C_ID_VOID_COL_NAME = "C_ID_void";
  const string C_REF_GENOME_ID_RLE_COL_NAME = "C_RG_ID_rle";

  const string RB_ID_VOID_COL_NAME = "RB_ID_void";
  const string RB_BASE_VALUE_BITPACKED_COL_NAME = "RB_BASE_VALUE_bitpacked";
  const string RB_POSITION_RLEDELTAONE_COL_NAME = "RB_POSITION_rledeltaone";
  const string RB_CONTIG_ID_RLE_COL_NAME = "RB_C_ID_rle";

  const string SG_ID_VOID_COL_NAME = "SG_ID_void";

  const string R_ID_VOID_COL_NAME = "R_ID_void";
  const string R_MAPPING_QUALITY_BITPACKED_COL_NAME = "R_MAPQ_bitpacked";
  const string R_MAPPING_QUALITY_RLE_COL_NAME = "R_MAPQ_rle";
  const string R_FLAG_BITPACKED_COL_NAME = "R_FLAG_bitpacked";
  const string R_FLAG_RLE_COL_NAME = "R_FLAG_rle";
  const string R_SAMPLE_GENOME_ID_RLE_COL_NAME = "R_SG_ID_rle";

  const string SB_ID_VOID_COL_NAME = "SB_ID_void";
  const string SB_BASE_VALUE_BITPACKED_COL_NAME = "SB_BASE_VALUE_bitpacked";
  const string SB_BASE_VALUE_REFERENCECOMPRESSED_COL_NAME =
      "SB_BASE_VALUE_referencecompressed";
  const string SB_BASE_CALL_QUALITY_RLE_COL_NAME = "SB_BASE_CALL_QUALITY_rle";
  const string SB_BASE_CALL_QUALITY_BITPACKED_COL_NAME =
      "SB_BASE_CALL_QUALITY_bitpacked";
  const string SB_INSERT_OFFSET_RLE_COL_NAME = "SB_INSERT_OFFSET_rle";
  const string SB_READ_ID_RLE_COL_NAME = "SB_READ_ID_rle";
  const string SB_RB_ID_RLEDELTAONE_COL_NAME = "SB_RB_ID_rledeltaone";

  class Storage_Experiments_Importer : public GenomeDataImporter {
   private:
    OIDTypedColumnPtr rg_id_column;
    OIDTypedColumnPtr rg_id_void_column;
    StringTypedColumnPtr rg_name_column;

    OIDTypedColumnPtr c_id_column;
    OIDTypedColumnPtr c_id_void_column;
    StringTypedColumnPtr c_name_column;
    OIDTypedColumnPtr c_rg_id_column;
    OIDTypedColumnPtr c_rg_id_rle_column;

    OIDTypedColumnPtr rb_id_column;
    OIDTypedColumnPtr rb_id_void_column;
    OIDTypedColumnPtr rb_c_id_column;
    OIDTypedColumnPtr rb_c_id_rle_column;
    IntTypedColumnPtr rb_pos_column;
    IntTypedColumnPtr rb_pos_rledelta_column;
    StringTypedColumnPtr rb_base_column;
    StringTypedColumnPtr rb_base_bitpacked_column;

    OIDTypedColumnPtr sg_id_column;
    OIDTypedColumnPtr sg_id_void_column;
    StringTypedColumnPtr sg_name_column;

    OIDTypedColumnPtr r_id_column;
    OIDTypedColumnPtr r_id_void_column;
    StringTypedColumnPtr r_qname_column;
    OIDTypedColumnPtr r_sg_id_column;
    OIDTypedColumnPtr r_sg_id_rle_column;
    IntTypedColumnPtr r_flag_column;
    IntTypedColumnPtr r_flag_rle_column;
    IntTypedColumnPtr r_flag_bitpacked_column;
    IntTypedColumnPtr r_map_qual_column;
    IntTypedColumnPtr r_map_qual_rle_column;
    IntTypedColumnPtr r_map_qual_bitpacked_column;
    OIDTypedColumnPtr r_mate_id_column;

    OIDTypedColumnPtr sb_id_column;
    OIDTypedColumnPtr sb_id_void_column;
    OIDTypedColumnPtr sb_rb_id_column;
    OIDTypedColumnPtr sb_rb_id_rledelta_column;
    OIDTypedColumnPtr sb_r_id_column;
    OIDTypedColumnPtr sb_r_id_rle_column;
    StringTypedColumnPtr sb_base_column;
    StringTypedColumnPtr sb_base_bitpacked_column;
    StringTypedColumnPtr sb_base_reference_based_column;
    IntTypedColumnPtr sb_insert_column;
    IntTypedColumnPtr sb_insert_rle_column;
    IntTypedColumnPtr sb_qual_column;
    IntTypedColumnPtr sb_qual_rle_column;
    IntTypedColumnPtr sb_qual_bitpacked_column;

    bool _createDatabase();

    bool _importReferenceGenomeData(const string reference_genome_name,
                                    const string path_to_fasta_file);

    bool _importAlignment(TID sample_genome_id,
                          TID id_of_first_reference_base_in_contig,
                          bam1_t *alignment, const bam1_core_t c, TID read_id,
                          TID mate_id, TID &sample_base_id,
                          cigar_conversion converted_cigar);

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

   public:
    Storage_Experiments_Importer(ostream &, bool verbose = true);

    bool importReferenceGenomeData(const string path_to_fasta_file,
                                   const string reference_genome_name);

    bool importSampleGenomeData(const string path_to_sambam_file,
                                const string reference_genome_name,
                                const string sample_genome_name);
  };

}  // end namespace CoGaDB

#endif  // BAM_FOUND

#endif  // STORAGE_EXPERIMENTS_IMPORTER_FILES_HPP
