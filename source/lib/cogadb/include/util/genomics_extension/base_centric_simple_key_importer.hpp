//
// Created by Sebastian Dorok on 06.07.15.
//

//#pragma once

#ifndef BASE_CENTRIC_SIMPLE_KEY_IMPORTER_FILES_HPP
#define BASE_CENTRIC_SIMPLE_KEY_IMPORTER_FILES_HPP

#ifdef BAM_FOUND

#include <compression/bitpacked_dictionary_compressed_column.hpp>
#include <compression/bitpacked_dictionary_compressed_column_base_values.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <compression/dictionary_compressed_column_base_values.hpp>
#include <compression/reference_based_compressed_column.hpp>
#include <compression/rle_compressed_column.hpp>
#include <compression/rle_compressed_column_with_prefix_counts.hpp>
#include <compression/rle_delta_one_compressed_column_int.hpp>
#include <compression/rle_delta_one_compressed_column_int_with_prefix_counts.hpp>
#include <compression/void_compressed_column_int.hpp>
#include <core/table.hpp>
#include <parser/client.hpp>
#include <query_processing/query_processor.hpp>
#include <util/genomics_extension/genome_data_importer.hpp>

#include <samtools/bam.h>
#include <samtools/sam.h>

/* some CIGAR related macros or only available since samtools 0.1.19 so define
 * them here */
#ifndef bam_cigar_op
#define bam_cigar_op(c) ((c)&BAM_CIGAR_MASK)
#endif
#ifndef bam_cigar_oplen
#define bam_cigar_oplen(c) ((c) >> BAM_CIGAR_SHIFT)
#endif

namespace CoGaDB {
  using namespace std;

  // table and column names
  const string RG_TBL_NAME = "REFERENCE_GENOME";
  const string RG_ID_COL_NAME = "RG_ID";
  const string RG_NAME_COL_NAME = "RG_NAME";

  const string C_TBL_NAME = "CONTIG";
  const string C_ID_COL_NAME = "C_ID";
  const string C_NAME_COL_NAME = "C_NAME";
  const string C_REF_GENOME_ID_COL_NAME = "C_RG_ID";

  const string RB_TBL_NAME = "REFERENCE_BASE";
  const string RB_ID_COL_NAME = "RB_ID";
  const string RB_BASE_VALUE_COL_NAME = "RB_BASE_VALUE";
  const string RB_POSITION_COL_NAME = "RB_POSITION";
  const string RB_CONTIG_ID_COL_NAME = "RB_C_ID";

  const string SG_TBL_NAME = "SAMPLE_GENOME";
  const string SG_ID_COL_NAME = "SG_ID";
  const string SG_NAME_COL_NAME = "SG_NAME";

  const string R_TBL_NAME = "READ";
  const string R_ID_COL_NAME = "R_ID";
  const string R_QNAME_COL_NAME = "R_QNAME";
  const string R_MAPPING_QUALITY_COL_NAME = "R_MAPQ";
  const string R_FLAG_COL_NAME = "R_FLAG";
  const string R_SAMPLE_GENOME_ID_COL_NAME = "R_SG_ID";
  const string R_MATE_ID_COL_NAME = "R_MATE_ID";

  const string SB_TBL_NAME = "SAMPLE_BASE";
  const string SB_ID_COL_NAME = "SB_ID";
  const string SB_BASE_VALUE_COL_NAME = "SB_BASE_VALUE";
  const string SB_BASE_CALL_QUALITY_COL_NAME = "SB_BASE_CALL_QUALITY";
  const string SB_INSERT_OFFSET_COL_NAME = "SB_INSERT_OFFSET";
  const string SB_READ_ID_COL_NAME = "SB_READ_ID";
  const string SB_RB_ID_COL_NAME = "SB_RB_ID";

  class BaseCentric_SimpleKey_Importer : public GenomeDataImporter {
   private:
    OIDTypedColumnPtr rg_id_column;
    StringTypedColumnPtr rg_name_column;

    OIDTypedColumnPtr c_id_column;
    StringTypedColumnPtr c_name_column;
    OIDTypedColumnPtr c_rg_id_column;

    OIDTypedColumnPtr rb_id_column;
    OIDTypedColumnPtr rb_c_id_column;
    IntTypedColumnPtr rb_pos_column;
    StringTypedColumnPtr rb_base_column;

    OIDTypedColumnPtr sg_id_column;
    StringTypedColumnPtr sg_name_column;

    OIDTypedColumnPtr r_id_column;
    StringTypedColumnPtr r_qname_column;
    OIDTypedColumnPtr r_sg_id_column;
    IntTypedColumnPtr r_flag_column;
    IntTypedColumnPtr r_map_qual_column;
    OIDTypedColumnPtr r_mate_id_column;

    OIDTypedColumnPtr sb_id_column;
    OIDTypedColumnPtr sb_rb_id_column;
    OIDTypedColumnPtr sb_r_id_column;
    StringTypedColumnPtr sb_base_column;
    IntTypedColumnPtr sb_insert_column;
    IntTypedColumnPtr sb_qual_column;

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
    BaseCentric_SimpleKey_Importer(ostream &, bool compress_data = true,
                                   bool verbose = true);

    bool importReferenceGenomeData(const string path_to_fasta_file,
                                   const string reference_genome_name);

    bool importSampleGenomeData(const string path_to_sambam_file,
                                const string reference_genome_name,
                                const string sample_genome_name);
  };

}  // end namespace CoGaDB

#endif  // BAM_FOUND

#endif  // BASE_CENTRIC_SIMPLE_KEY_IMPORTER_FILES_HPP
