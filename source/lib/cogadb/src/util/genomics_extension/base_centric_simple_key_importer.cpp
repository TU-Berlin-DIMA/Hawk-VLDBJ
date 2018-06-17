//
// Created by Sebastian Dorok on 06.07.15.
//

#include <core/global_definitions.hpp>

#ifdef BAM_FOUND

#include <util/genomics_extension/base_centric_simple_key_importer.hpp>
#include "../../../include/core/variable_manager.hpp"

namespace CoGaDB {

// taken from stdint.h
#define UINT64MAX 18446744073709551615UL

using namespace std;

bool BaseCentric_SimpleKey_Importer::_createDatabase() {
  out << "Creating genome database" << endl;
  out << "Type: base centric with simple keys" << endl;
  out << "Compression: " << (compress_data ? "true" : "false") << endl;

  if (this->compress_data) {
    // #########################
    // # reference genome data #
    // #########################
    // reference genome
    rg_id_column = OIDTypedColumnPtr(
        new VoidCompressedColumnNumber<TID>(RG_ID_COL_NAME, OID));
    rg_name_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(RG_NAME_COL_NAME, VARCHAR));
    // contig
    c_id_column = OIDTypedColumnPtr(
        new VoidCompressedColumnNumber<TID>(C_ID_COL_NAME, OID));
    c_name_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(C_NAME_COL_NAME, VARCHAR));
    c_rg_id_column = OIDTypedColumnPtr(
        new RLECompressedColumn<TID>(C_REF_GENOME_ID_COL_NAME, OID));
    // reference base
    rb_id_column = OIDTypedColumnPtr(
        new VoidCompressedColumnNumber<TID>(RB_ID_COL_NAME, OID));
    // base values are aggressively compressed - 3 bit
    std::map<string, uint32_t> dictionary;
    // 3 bit allows for 8 values
    vector<string> reverse_look_up_vector(8);
    string base_values[6] = {"A", "C", "G", "T", "X", "N"};
    string value = "";
    uint32_t key = 0;
    for (int i = 0; i < 6; i++) {
      value = base_values[i];
      key = (uint32_t)*value.c_str() & 7;
      dictionary.insert(make_pair(value, key));
      reverse_look_up_vector[key] = value;
    }
    rb_base_column = StringTypedColumnPtr(
        new BitPackedDictionaryCompressedColumnForBaseValues<std::string>(
            RB_BASE_VALUE_COL_NAME, VARCHAR, dictionary, reverse_look_up_vector,
            3));
    // rb_pos_column = IntTypedColumnPtr(new
    // RLEDeltaOneCompressedColumnNumber<int>(RB_POSITION_COL_NAME, INT));
    rb_pos_column = IntTypedColumnPtr(
        new RLEDeltaOneCompressedColumnNumberWithPrefixCounts<int>(
            RB_POSITION_COL_NAME, INT));
    //            rb_pos_column = IntTypedColumnPtr(new
    //            Column<int>(RB_POSITION_COL_NAME, INT));
    rb_c_id_column = OIDTypedColumnPtr(
        new RLECompressedColumn<TID>(RB_CONTIG_ID_COL_NAME, OID));

    // #################
    // # sample genome #
    // #################
    // sample genome
    sg_id_column = OIDTypedColumnPtr(
        new VoidCompressedColumnNumber<TID>(SG_ID_COL_NAME, OID));
    sg_name_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(SG_NAME_COL_NAME, VARCHAR));
    // read
    r_id_column = OIDTypedColumnPtr(
        new VoidCompressedColumnNumber<TID>(R_ID_COL_NAME, OID));
    r_qname_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(R_QNAME_COL_NAME, VARCHAR));
    r_sg_id_column = OIDTypedColumnPtr(
        new RLECompressedColumn<TID>(R_SAMPLE_GENOME_ID_COL_NAME, OID));
    r_map_qual_column = IntTypedColumnPtr(
        new RLECompressedColumn<int>(R_MAPPING_QUALITY_COL_NAME, INT));
    r_flag_column = IntTypedColumnPtr(
        new BitPackedDictionaryCompressedColumn<int>(R_FLAG_COL_NAME, INT, 16));
    // FIXME mate information - no compression yet
    r_mate_id_column =
        OIDTypedColumnPtr(new Column<TID>(R_MATE_ID_COL_NAME, OID));
    // sample base
    sb_id_column = OIDTypedColumnPtr(
        new VoidCompressedColumnNumber<TID>(SB_ID_COL_NAME, OID));
    sb_rb_id_column = OIDTypedColumnPtr(
        new RLEDeltaOneCompressedColumnNumberWithPrefixCounts<TID>(
            SB_RB_ID_COL_NAME, OID));
    //            sb_rb_id_column = OIDTypedColumnPtr(new
    //            Column<TID>(SB_RB_ID_COL_NAME, OID));
    sb_r_id_column = OIDTypedColumnPtr(
        new RLECompressedColumn<TID>(SB_READ_ID_COL_NAME, OID));
    // sb_base_column = StringTypedColumnPtr(
    //        new
    //        BitPackedDictionaryCompressedColumnForBaseValues<std::string>(SB_BASE_VALUE_COL_NAME,
    //        VARCHAR,
    //                                                                          dictionary,
    //                                                                          reverse_look_up_vector, 3));
    sb_base_column =
        StringTypedColumnPtr(new ReferenceBasedCompressedColumn<std::string>(
            SB_BASE_VALUE_COL_NAME, VARCHAR, SB_TBL_NAME, SB_RB_ID_COL_NAME,
            RB_TBL_NAME, RB_BASE_VALUE_COL_NAME));
    // sb_insert_column = IntTypedColumnPtr(new
    // RLECompressedColumn<int>(SB_INSERT_OFFSET_COL_NAME, INT));
    sb_insert_column =
        IntTypedColumnPtr(new RLECompressedColumnWithPrefixCounts<int>(
            SB_INSERT_OFFSET_COL_NAME, INT));
    sb_qual_column =
        IntTypedColumnPtr(new BitPackedDictionaryCompressedColumn<int>(
            SB_BASE_CALL_QUALITY_COL_NAME, INT, 8));
  } else {
    // #########################
    // # reference genome data #
    // #########################
    // reference genome
    rg_id_column = OIDTypedColumnPtr(new Column<TID>(RG_ID_COL_NAME, OID));
    rg_name_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(RG_NAME_COL_NAME, VARCHAR));
    // contig
    c_id_column = OIDTypedColumnPtr(new Column<TID>(C_ID_COL_NAME, OID));
    c_name_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(C_NAME_COL_NAME, VARCHAR));
    c_rg_id_column =
        OIDTypedColumnPtr(new Column<TID>(C_REF_GENOME_ID_COL_NAME, OID));
    // reference base
    rb_id_column = OIDTypedColumnPtr(new Column<TID>(RB_ID_COL_NAME, OID));

    std::map<string, uint32_t> dictionary;
    // ascii code of 'Z'
    vector<string> reverse_look_up_vector(90);
    string base_values[6] = {"A", "C", "G", "T", "X", "N"};
    string value = "";
    uint32_t key = 0;
    for (int i = 0; i < 6; i++) {
      value = base_values[i];
      key = (uint32_t)*value.c_str();
      dictionary.insert(make_pair(value, key));
      reverse_look_up_vector[key] = value;
    }
    rb_base_column = StringTypedColumnPtr(
        new DictionaryCompressedColumnForBaseValues<std::string>(
            RB_BASE_VALUE_COL_NAME, VARCHAR, dictionary,
            reverse_look_up_vector));
    rb_pos_column =
        IntTypedColumnPtr(new Column<int>(RB_POSITION_COL_NAME, INT));
    rb_c_id_column =
        OIDTypedColumnPtr(new Column<TID>(RB_CONTIG_ID_COL_NAME, OID));

    // #################
    // # sample genome #
    // #################
    // sample genome
    sg_id_column = OIDTypedColumnPtr(new Column<TID>(SG_ID_COL_NAME, OID));
    sg_name_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(SG_NAME_COL_NAME, VARCHAR));
    // read
    r_id_column = OIDTypedColumnPtr(new Column<TID>(R_ID_COL_NAME, OID));
    r_qname_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(R_QNAME_COL_NAME, VARCHAR));
    r_sg_id_column =
        OIDTypedColumnPtr(new Column<TID>(R_SAMPLE_GENOME_ID_COL_NAME, OID));
    r_map_qual_column =
        IntTypedColumnPtr(new Column<int>(R_MAPPING_QUALITY_COL_NAME, INT));
    r_flag_column = IntTypedColumnPtr(new Column<int>(R_FLAG_COL_NAME, INT));
    // FIXME mate information - no compression yet
    r_mate_id_column =
        OIDTypedColumnPtr(new Column<TID>(R_MATE_ID_COL_NAME, OID));
    // sample base
    sb_id_column = OIDTypedColumnPtr(new Column<TID>(SB_ID_COL_NAME, OID));
    sb_rb_id_column =
        OIDTypedColumnPtr(new Column<TID>(SB_RB_ID_COL_NAME, OID));
    sb_r_id_column =
        OIDTypedColumnPtr(new Column<TID>(SB_READ_ID_COL_NAME, OID));
    // sb_base = ColumnPtr(new
    // DictionaryCompressedColumn<std::string>(SB_BASE_VALUE_COL_NAME,
    // VARCHAR));
    sb_base_column = StringTypedColumnPtr(
        new DictionaryCompressedColumnForBaseValues<std::string>(
            SB_BASE_VALUE_COL_NAME, VARCHAR, dictionary,
            reverse_look_up_vector));
    sb_insert_column =
        IntTypedColumnPtr(new Column<int>(SB_INSERT_OFFSET_COL_NAME, INT));
    sb_qual_column =
        IntTypedColumnPtr(new Column<int>(SB_BASE_CALL_QUALITY_COL_NAME, INT));
  }

  // Setting up tables inclusive primary and foreign keys
  vector<ColumnPtr> ref_genome_cols;
  vector<ColumnPtr> contig_cols;
  vector<ColumnPtr> ref_base_cols;
  vector<ColumnPtr> sample_genome_cols;
  vector<ColumnPtr> read_cols;
  vector<ColumnPtr> sample_base_cols;
  // reference genome
  out << "Creating " << RG_TBL_NAME << " table ..." << endl;
  ref_genome_cols.push_back(rg_id_column);
  ref_genome_cols.push_back(rg_name_column);
  refGenomeTable = TablePtr(new Table(RG_TBL_NAME, ref_genome_cols));
  if (!refGenomeTable->setPrimaryKeyConstraint(RG_ID_COL_NAME)) {
    COGADB_ERROR("Failed to set Primary Key Constraint!", "");
    return false;
  }
  addToGlobalTableList(refGenomeTable);
  // contig
  out << "Creating " << C_TBL_NAME << " table ..." << endl;
  contig_cols.push_back(c_id_column);
  contig_cols.push_back(c_name_column);
  contig_cols.push_back(c_rg_id_column);
  contigTable = TablePtr(new Table(C_TBL_NAME, contig_cols));
  if (!contigTable->setPrimaryKeyConstraint(C_ID_COL_NAME)) {
    COGADB_ERROR("Failed to set Primary Key Constraint!", "");
    return false;
  }
  if (!contigTable->setForeignKeyConstraint(C_REF_GENOME_ID_COL_NAME,
                                            RG_ID_COL_NAME, RG_TBL_NAME)) {
    COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
    return false;
  }
  addToGlobalTableList(contigTable);
  // reference base
  out << "Creating " << RB_TBL_NAME << " table ..." << endl;
  ref_base_cols.push_back(rb_id_column);
  ref_base_cols.push_back(rb_base_column);
  ref_base_cols.push_back(rb_pos_column);
  ref_base_cols.push_back(rb_c_id_column);
  refBaseTable = TablePtr(new Table(RB_TBL_NAME, ref_base_cols));
  if (!refBaseTable->setPrimaryKeyConstraint(RB_ID_COL_NAME)) {
    COGADB_ERROR("Failed to set Primary Key Constraint!", "");
    return false;
  }
  if (!refBaseTable->setForeignKeyConstraint(RB_CONTIG_ID_COL_NAME,
                                             C_ID_COL_NAME, C_TBL_NAME)) {
    COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
    return false;
  }
  addToGlobalTableList(refBaseTable);

  // sample genome
  out << "Creating " << SG_TBL_NAME << " table ..." << endl;
  sample_genome_cols.push_back(sg_id_column);
  sample_genome_cols.push_back(sg_name_column);
  sampleGenomeTable = TablePtr(new Table(SG_TBL_NAME, sample_genome_cols));
  if (!sampleGenomeTable->setPrimaryKeyConstraint(SG_ID_COL_NAME)) {
    COGADB_ERROR("Failed to set Primary Key Constraint!", "");
    return false;
  }
  addToGlobalTableList(sampleGenomeTable);
  // read
  out << "Creating " << R_TBL_NAME << " table ..." << endl;
  read_cols.push_back(r_id_column);
  read_cols.push_back(r_qname_column);
  read_cols.push_back(r_sg_id_column);
  read_cols.push_back(r_map_qual_column);
  read_cols.push_back(r_flag_column);
  read_cols.push_back(r_mate_id_column);
  readTable = TablePtr(new Table(R_TBL_NAME, read_cols));
  if (!readTable->setPrimaryKeyConstraint(R_ID_COL_NAME)) {
    COGADB_ERROR("Failed to set Primary Key Constraint!", "");
    return false;
  }
  if (!readTable->setForeignKeyConstraint(R_SAMPLE_GENOME_ID_COL_NAME,
                                          SG_ID_COL_NAME, SG_TBL_NAME)) {
    COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
    return false;
  }
  // add read table before we can set a foreign key
  addToGlobalTableList(readTable);
  if (!readTable->setForeignKeyConstraint(R_MATE_ID_COL_NAME, R_ID_COL_NAME,
                                          R_TBL_NAME)) {
    COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
    return false;
  }
  // sample base
  out << "Creating " << SB_TBL_NAME << " table ..." << endl;
  sample_base_cols.push_back(sb_id_column);
  sample_base_cols.push_back(sb_rb_id_column);
  sample_base_cols.push_back(sb_r_id_column);
  sample_base_cols.push_back(sb_base_column);
  sample_base_cols.push_back(sb_insert_column);
  sample_base_cols.push_back(sb_qual_column);
  sampleBaseTable = TablePtr(new Table(SB_TBL_NAME, sample_base_cols));
  if (!sampleBaseTable->setPrimaryKeyConstraint(SB_ID_COL_NAME)) {
    COGADB_ERROR("Failed to set Primary Key Constraint!", "");
    return false;
  }
  if (!sampleBaseTable->setForeignKeyConstraint(SB_READ_ID_COL_NAME,
                                                R_ID_COL_NAME, R_TBL_NAME)) {
    COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
    return false;
  }
  if (!sampleBaseTable->setForeignKeyConstraint(SB_RB_ID_COL_NAME,
                                                RB_ID_COL_NAME, RB_TBL_NAME)) {
    COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
    return false;
  }
  addToGlobalTableList(sampleBaseTable);

  if (sb_base_column->getColumnType() == REFERENCE_BASED_COMPRESSED) {
    boost::shared_ptr<ReferenceBasedCompressedColumn<string> >
        sb_base_column_typed = boost::dynamic_pointer_cast<
            ReferenceBasedCompressedColumn<string> >(sb_base_column);
    assert(sb_base_column_typed != NULL);
    sb_base_column_typed->initReferenceColumnPointers();
  }

  // Store the tables on disk
  storeTable(refGenomeTable);
  storeTable(contigTable);
  storeTable(refBaseTable);
  storeTable(sampleGenomeTable);
  storeTable(readTable);
  storeTable(sampleBaseTable);
  return true;
}

bool BaseCentric_SimpleKey_Importer::_initDatabaseConnection() {
  // check whether database exists
  vector<TablePtr> &tables = getGlobalTableList();
  int existing_tables = 0;
  for (std::vector<TablePtr>::const_iterator table = tables.begin();
       table != tables.end(); ++table) {
    if ((*table)->getName().compare(RG_TBL_NAME) == 0) {
      refGenomeTable = *table;
      existing_tables++;
      continue;
    }
    if ((*table)->getName().compare(C_TBL_NAME) == 0) {
      contigTable = *table;
      existing_tables++;
      continue;
    }
    if ((*table)->getName().compare(RB_TBL_NAME) == 0) {
      refBaseTable = *table;
      existing_tables++;
      continue;
    }
    if ((*table)->getName().compare(SG_TBL_NAME) == 0) {
      sampleGenomeTable = *table;
      existing_tables++;
      continue;
    }
    if ((*table)->getName().compare(R_TBL_NAME) == 0) {
      readTable = *table;
      existing_tables++;
      continue;
    }
    if ((*table)->getName().compare(SB_TBL_NAME) == 0) {
      sampleBaseTable = *table;
      existing_tables++;
      continue;
    }
  }
  if (existing_tables == 6) {
    rg_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        refGenomeTable->getColumnbyName(RG_ID_COL_NAME));
    assert(rg_id_column != NULL);
    rg_name_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        refGenomeTable->getColumnbyName(RG_NAME_COL_NAME));
    assert(rg_name_column != NULL);

    c_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        contigTable->getColumnbyName(C_ID_COL_NAME));
    assert(c_id_column != NULL);
    c_name_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        contigTable->getColumnbyName(C_NAME_COL_NAME));
    assert(c_name_column != NULL);
    c_rg_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        contigTable->getColumnbyName(C_REF_GENOME_ID_COL_NAME));
    assert(c_rg_id_column != NULL);

    rb_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        refBaseTable->getColumnbyName(RB_ID_COL_NAME));
    assert(rb_id_column != NULL);
    rb_c_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        refBaseTable->getColumnbyName(RB_CONTIG_ID_COL_NAME));
    assert(rb_c_id_column != NULL);
    rb_pos_column = boost::dynamic_pointer_cast<IntTypedColumnType>(
        refBaseTable->getColumnbyName(RB_POSITION_COL_NAME));
    assert(rb_pos_column != NULL);
    rb_base_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        refBaseTable->getColumnbyName(RB_BASE_VALUE_COL_NAME));

    sg_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        sampleGenomeTable->getColumnbyName(SG_ID_COL_NAME));
    assert(sg_id_column != NULL);
    sg_name_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        sampleGenomeTable->getColumnbyName(SG_NAME_COL_NAME));
    assert(sg_name_column != NULL);

    r_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        readTable->getColumnbyName(R_ID_COL_NAME));
    assert(r_id_column != NULL);
    r_qname_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        readTable->getColumnbyName(R_QNAME_COL_NAME));
    assert(r_qname_column != NULL);
    r_sg_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        readTable->getColumnbyName(R_SAMPLE_GENOME_ID_COL_NAME));
    assert(r_sg_id_column != NULL);
    r_flag_column = boost::dynamic_pointer_cast<IntTypedColumnType>(
        readTable->getColumnbyName(R_FLAG_COL_NAME));
    assert(r_flag_column != NULL);
    r_map_qual_column = boost::dynamic_pointer_cast<IntTypedColumnType>(
        readTable->getColumnbyName(R_MAPPING_QUALITY_COL_NAME));
    assert(r_map_qual_column != NULL);
    r_mate_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        readTable->getColumnbyName(R_MATE_ID_COL_NAME));
    assert(r_mate_id_column != NULL);

    sb_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        sampleBaseTable->getColumnbyName(SB_ID_COL_NAME));
    assert(sb_id_column != NULL);
    sb_rb_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        sampleBaseTable->getColumnbyName(SB_RB_ID_COL_NAME));
    assert(sb_rb_id_column != NULL);
    sb_r_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        sampleBaseTable->getColumnbyName(SB_READ_ID_COL_NAME));
    assert(sb_r_id_column != NULL);
    sb_base_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        sampleBaseTable->getColumnbyName(SB_BASE_VALUE_COL_NAME));
    if (sb_base_column->getColumnType() == REFERENCE_BASED_COMPRESSED) {
      boost::shared_ptr<ReferenceBasedCompressedColumn<string> >
          sb_base_column_typed = boost::dynamic_pointer_cast<
              ReferenceBasedCompressedColumn<string> >(sb_base_column);
      assert(sb_base_column_typed != NULL);
      sb_base_column_typed->initReferenceColumnPointers();
    }
    assert(sb_base_column != NULL);
    sb_insert_column = boost::dynamic_pointer_cast<IntTypedColumnType>(
        sampleBaseTable->getColumnbyName(SB_INSERT_OFFSET_COL_NAME));
    assert(sb_insert_column != NULL);
    sb_qual_column = boost::dynamic_pointer_cast<IntTypedColumnType>(
        sampleBaseTable->getColumnbyName(SB_BASE_CALL_QUALITY_COL_NAME));
    assert(sb_qual_column != NULL);
    return true;
  } else if (existing_tables == 0) {
    // none of the required tables exists -> create database
    return this->_createDatabase();
  } else {
    this->out << "ERROR: There is at least one existing table. Database "
                 "corrupted? Or other schema present? Aborting ..."
              << std::endl;
    return false;
  }
}

BaseCentric_SimpleKey_Importer::BaseCentric_SimpleKey_Importer(
    ostream &_out, bool _compress_data, bool _verbose)
    : GenomeDataImporter(_out, _compress_data, _verbose) {
  this->_initDatabaseConnection();
}

bool BaseCentric_SimpleKey_Importer::_importReferenceGenomeData(
    const string reference_genome_name, const string path_to_fasta_file) {
  Timestamp start = getTimestamp();
  Timestamp end_preparation = 0;
  Timestamp end_importing = 0;
  Timestamp end_storing = 0;

  // initiate id
  TID reference_genome_id = refGenomeTable->getNumberofRows();
  TID contigId = contigTable->getNumberofRows() - 1;
  TID rb_id = refBaseTable->getNumberofRows();

  rg_id_column->insert(reference_genome_id);
  rg_name_column->insert(reference_genome_name);

  // Insert data from FASTA file
  string fastaLine;
  int ref_base_position = 0;
  ifstream fastaFile(path_to_fasta_file.c_str());

  // measure file length
  fastaFile.seekg(0, fastaFile.end);
  uint64_t lengthOverall = fastaFile.tellg();
  uint64_t printProgressStep = ceil(0.001 * lengthOverall);
  uint64_t printProgressCounter = printProgressStep;
  fastaFile.seekg(0, fastaFile.beg);
  uint64_t length = 0;

  end_preparation = getTimestamp() - start;
  start = getTimestamp();

  if (fastaFile.is_open()) {
    if (this->verbose) {
      _drawProgress(lengthOverall, length, 50, out);
    }
    while (getline(fastaFile, fastaLine)) {
      if (fastaLine[0] == '>') {
        contigId++;
        ref_base_position = 0;
        /* found a new reference sequence */
        // Tuple contigTuple;
        // Search for first while space in the contig name to cut a meaningful
        // contig name.
        // If the name does not contain a white the return value is
        // string::npos, which means the following
        // substr cuts returns the complete contig name.
        size_t firstWhiteSpace = fastaLine.find(" ");
        if (firstWhiteSpace != string::npos) {
          firstWhiteSpace--;
        }
        c_id_column->insert(contigId);
        c_name_column->insert(fastaLine.substr(1, firstWhiteSpace));
        c_rg_id_column->insert(reference_genome_id);
      } else {
        /* bases for reference found */
        int i = 0;
        while (fastaLine[i] != '\n' && fastaLine[i] != '\0') {
          // Tuple refBaseTuple;
          rb_id_column->insert(rb_id);
          // refBaseTuple.push_back(fastaLine.substr(i, 1));
          rb_base_column->insert(std::string(1, fastaLine[i]));
          rb_pos_column->insert(ref_base_position);
          rb_c_id_column->insert(contigId);
          i++;
          ref_base_position++;
          rb_id++;
        }
      }
      length = fastaFile.tellg();
      if (this->verbose && length > printProgressCounter) {
        _drawProgress(lengthOverall, length, 50, out);
        printProgressCounter += printProgressStep;
      }
    }
    if (this->verbose) {
      _drawProgress(lengthOverall, lengthOverall, 50, out, true);
    }
    fastaFile.close();

    refGenomeTable->setNumberofRows(reference_genome_id + 1);
    contigTable->setNumberofRows(contigId + 1);
    refBaseTable->setNumberofRows(rb_id);
  } else {
    out << "ERROR: Unable to open file: '" << path_to_fasta_file << "'" << endl;
    return false;
  }

  end_importing = getTimestamp() - start;
  start = getTimestamp();

  // Store data
  storeTable(refGenomeTable);
  storeTable(contigTable);
  storeTable(refBaseTable);
  end_storing = getTimestamp() - start;

  out << "Time statistics:" << endl;
  out << fixed << showpoint << setprecision(2);
  std::pair<double, string> time =
      _getTimeAsHumanReadableString(end_preparation);
  out << "Time to PREPARE import: " << time.first << time.second << " ("
      << end_preparation << "ns)" << endl;
  time = _getTimeAsHumanReadableString(end_importing);
  out << "Time to IMPORT reference data: " << time.first << time.second << " ("
      << end_importing << "ns)" << endl;
  time = _getTimeAsHumanReadableString(end_storing);
  out << "Time to STORE data on disk: " << time.first << time.second << " ("
      << end_storing << "ns)" << endl;

  return true;
}

bool BaseCentric_SimpleKey_Importer::importReferenceGenomeData(
    const string path_to_fasta_file, const string reference_genome_name) {
  // importing reference genome data
  this->out << "Importing reference genome data" << endl;
  this->out << "FASTA file: " << path_to_fasta_file << endl;
  this->out << "Reference genome name: " << reference_genome_name << endl;

  bool success = false;
  success = this->_importReferenceGenomeData(reference_genome_name,
                                             path_to_fasta_file);
  if (success) {
    out << " SUCCESS!" << endl;
    out << "Following tables were imported:" << endl;
    out << refGenomeTable->getName() << "(" << refGenomeTable->getNumberofRows()
        << " rows, " << refGenomeTable->getSizeinBytes() << " bytes)" << endl;
    out << contigTable->getName() << "(" << contigTable->getNumberofRows()
        << " rows, " << contigTable->getSizeinBytes() << " bytes)" << endl;
    out << refBaseTable->getName() << "(" << refBaseTable->getNumberofRows()
        << " rows, " << refBaseTable->getSizeinBytes() << " bytes)" << endl;
  } else {
    out << " ERROR!" << endl;
  }
  return success;
}

/**
* bam1_t* alignment ... to convert
* int* numberOfUnmappedReads ... for statistics
* int* numberOfReads ... for statistics
*/
bool BaseCentric_SimpleKey_Importer::_importAlignment(
    TID sample_genome_id, TID id_of_first_reference_base_in_contig,
    bam1_t *alignment, const bam1_core_t c, TID read_id, TID mate_id,
    TID &sample_base_id, cigar_conversion converted_cigar) {
  int i;
  string base_value = "";
  int32_t base_call_quality = -1;

  int32_t pos = c.pos;

  int avg_base_call_quality = 0;

  uint8_t *baseSequence, *qualityValues;

  /* extract alignment and quality values */
  baseSequence = bam1_seq(alignment);
  qualityValues = bam1_qual(alignment);

  /* read_id, mapping quality, description */
  r_id_column->insert(read_id);
  r_qname_column->insert(string(bam1_qname(alignment)));
  r_sg_id_column->insert(sample_genome_id);
  r_flag_column->insert((int)c.flag);
  r_map_qual_column->insert((int)c.qual);
  r_mate_id_column->insert(mate_id);

  // assert(sb_base_column->getColumnType() == REFERENCE_BASED_COMPRESSED);
  if (sb_base_column->getColumnType() == REFERENCE_BASED_COMPRESSED) {
    boost::shared_ptr<ReferenceBasedCompressedColumn<string> >
        sb_base_value_referenced = boost::dynamic_pointer_cast<
            ReferenceBasedCompressedColumn<string> >(sb_base_column);
    assert(sb_base_value_referenced != NULL);

    /* iterate through the sequence */
    for (i = 0; i < (c.l_qseq - (converted_cigar.start_offset +
                                 converted_cigar.end_offset));
         ++i) {
      base_value = "";
      base_value += bam_nt16_rev_table[bam1_seqi(
          baseSequence, (i + converted_cigar.start_offset))];
      if (qualityValues[0] == 0xff) {
        base_call_quality = '*';
      } else {
        base_call_quality =
            (char)(qualityValues[i + converted_cigar.start_offset]);
        avg_base_call_quality += base_call_quality;
      }
      TID sb_rb_id = id_of_first_reference_base_in_contig + pos +
                     (int)converted_cigar.positions[i];
      sb_id_column->insert(sample_base_id);
      sb_rb_id_column->insert(sb_rb_id);
      sb_r_id_column->insert(read_id);
      sb_base_value_referenced->insert((string)base_value, sb_rb_id);
      sb_insert_column->insert((int)converted_cigar.insert_offsets[i]);
      sb_qual_column->insert(base_call_quality);
      ++sample_base_id;
    }
    /* determine average base call quality for deleted bases */
    avg_base_call_quality = (avg_base_call_quality / (i + 1));
    /* iterate through deletions */
    for (i = 0; i < converted_cigar.del_end_offset; i++) {
      base_value = "X";
      TID sb_rb_id = id_of_first_reference_base_in_contig + pos +
                     (int)converted_cigar.deleted_positions[i];
      sb_id_column->insert(sample_base_id);
      sb_rb_id_column->insert(sb_rb_id);
      sb_r_id_column->insert(read_id);
      sb_base_value_referenced->insert((string)base_value, sb_rb_id);
      sb_insert_column->insert((int)converted_cigar.insert_offsets[i]);
      sb_qual_column->insert(avg_base_call_quality);
      ++sample_base_id;
    }
  } else {
    /* iterate through the sequence */
    for (i = 0; i < (c.l_qseq - (converted_cigar.start_offset +
                                 converted_cigar.end_offset));
         ++i) {
      base_value = "";
      base_value += bam_nt16_rev_table[bam1_seqi(
          baseSequence, (i + converted_cigar.start_offset))];
      if (qualityValues[0] == 0xff) {
        base_call_quality = '*';
      } else {
        base_call_quality =
            (char)(qualityValues[i + converted_cigar.start_offset]);
        avg_base_call_quality += base_call_quality;
      }
      TID sb_rb_id = id_of_first_reference_base_in_contig + pos +
                     (int)converted_cigar.positions[i];
      sb_id_column->insert(sample_base_id);
      sb_rb_id_column->insert(sb_rb_id);
      sb_r_id_column->insert(read_id);
      sb_base_column->insert((string)base_value);
      sb_insert_column->insert((int)converted_cigar.insert_offsets[i]);
      sb_qual_column->insert(base_call_quality);
      ++sample_base_id;
    }
    /* determine average base call quality for deleted bases */
    avg_base_call_quality = (avg_base_call_quality / (i + 1));
    /* iterate through deletions */
    for (i = 0; i < converted_cigar.del_end_offset; i++) {
      base_value = "X";
      TID sb_rb_id = id_of_first_reference_base_in_contig + pos +
                     (int)converted_cigar.deleted_positions[i];
      sb_id_column->insert(sample_base_id);
      sb_rb_id_column->insert(sb_rb_id);
      sb_r_id_column->insert(read_id);
      sb_base_column->insert((string)base_value);
      sb_insert_column->insert((int)converted_cigar.insert_offsets[i]);
      sb_qual_column->insert(avg_base_call_quality);
      ++sample_base_id;
    }
  }

  return true;
}

bool BaseCentric_SimpleKey_Importer::_importSampleGenomeData(
    string sample_genome_name, TID reference_genome_id,
    string path_to_sam_bam_file, TID &numberOfUnmappedReads,
    TID &numberOfReads) {
  Timestamp start = getTimestamp();
  Timestamp end_preparation = 0;
  Timestamp end_counting = 0;
  Timestamp end_importing = 0;
  Timestamp end_storing = 0;

  // intialize ids
  TID sample_genome_id = sampleGenomeTable->getNumberofRows();
  TID read_id = readTable->getNumberofRows();
  TID sample_base_id = sampleBaseTable->getNumberofRows();

  sg_id_column->insert(sample_genome_id);
  sg_name_column->insert(sample_genome_name);

  // Prepare import information: Contig names, their database ids and the id of
  // the first reference within the contig (min(id))
  // The corresponding SQL query: select c_name, c_id, min(rb_id) from contig
  // join reference_base on c_id = rb_c_id
  // where c_rg_id = <reference_genome_id> group by c_name, c_id;
  map<string, contig_ids> contigName2IdMap;
  {
    // do everything on the CPU
    ProcessorSpecification processorSpecification(hype::PD0);

    PositionListPtr contig_TIDs;
    // where c_rg_id = 0
    contig_TIDs = c_rg_id_column->selection(
        SelectionParam(processorSpecification, ValueConstantPredicate,
                       reference_genome_id, EQUAL));

    if (contig_TIDs->size() == 0) {
      out << "ERROR: Couldn't find contigs for " << reference_genome_id << "."
          << endl;
      return false;
    }

    ColumnPtr selected_contig_ids =
        c_id_column->gather(contig_TIDs, GatherParam(processorSpecification));
    ColumnPtr selected_contig_names =
        c_name_column->gather(contig_TIDs, GatherParam(processorSpecification));
    // contig join reference_base on c_id = rb_c_id - first TID list for
    // (modified) contig, second for rb
    PositionListPairPtr join_pair = selected_contig_ids->join(
        rb_c_id_column, JoinParam(processorSpecification, HASH_JOIN));
    // contig_id_column stores TIDs of contig -> first join list contains the
    // joined contig TIDs
    ColumnPtr joined_contig_ids = selected_contig_ids->gather(
        join_pair->first, GatherParam(processorSpecification));
    // gather contig names
    ColumnPtr joined_contig_names = selected_contig_names->gather(
        join_pair->first, GatherParam(processorSpecification));
    // gather rb_ids - TODO check for void is not necessarily required as we
    // know how the data is stored
    ColumnPtr joined_rb_ids;
    if (rb_id_column->getColumnType() == VOID_COMPRESSED_NUMBER) {
      // VOID COMPRESSED mainly works for primary key columns with incrementing
      // keys
      // PK FK join yields a result where every FK TID appears exactly once
      // Thus, a gather is not needed as the join already computes the required
      // TIDs that are also stored
      // in a void_compressed column like rb_id_column
      joined_rb_ids = join_pair->second;
      // boost::dynamic_pointer_cast<ColumnBase>();
      // assert(joined_rb_ids != NULL);
    } else {
      joined_rb_ids = rb_id_column->gather(join_pair->second,
                                           GatherParam(processorSpecification));
    }

    vector<ColumnPtr> grouping_columns;
    // we just have to use one column for grouping as the pairs are also
    // distinct
    grouping_columns.push_back(joined_contig_ids);
    // grouping_columns.push_back(joined_contig_names);
    ColumnGroupingKeysPtr grouping_keys =
        CDK::aggregation::computeColumnGroupingKeys(grouping_columns,
                                                    processorSpecification);
    // Result are two ColumnPtr: 1st contains TID of grouping columns to get
    // values!, 2nd aggregated values
    AggregationResult aggregation_result =
        joined_rb_ids->aggregateByGroupingKeys(
            grouping_keys,
            AggregationParam(processorSpecification, MIN,
                             HASH_BASED_AGGREGATION, "MIN_RB_ID", true));

    PositionListPtr agg_result_TIDs =
        boost::dynamic_pointer_cast<Column<TID> >(aggregation_result.first);
    assert(agg_result_TIDs != NULL);
    ColumnPtr grouped_contig_ids = joined_contig_ids->gather(
        agg_result_TIDs, GatherParam(processorSpecification));
    assert(grouped_contig_ids != NULL);
    ColumnPtr grouped_contig_names = joined_contig_names->gather(
        agg_result_TIDs, GatherParam(processorSpecification));
    assert(grouped_contig_names != NULL);

    boost::shared_ptr<Column<TID> > queried_contig_ids;
    boost::shared_ptr<Column<string> > queried_contig_names;
    boost::shared_ptr<Column<TID> > queried_aggregation_results;

    // decompress contig ids if necessary
    OIDTypedColumnPtr queried_contig_ids_typed =
        boost::dynamic_pointer_cast<OIDTypedColumnType>(grouped_contig_ids);
    assert(queried_contig_ids_typed != NULL);
    if (queried_contig_ids_typed->getColumnType() != PLAIN_MATERIALIZED) {
      queried_contig_ids = queried_contig_ids_typed->copyIntoDenseValueColumn(
          processorSpecification);
    } else {
      queried_contig_ids =
          boost::dynamic_pointer_cast<Column<TID> >(grouped_contig_ids);
      assert(queried_contig_ids != NULL);
    }

    StringTypedColumnPtr queried_contig_names_typed =
        boost::dynamic_pointer_cast<StringTypedColumnType>(
            grouped_contig_names);
    assert(queried_contig_names_typed != NULL);
    queried_contig_names = queried_contig_names_typed->copyIntoDenseValueColumn(
        processorSpecification);

    OIDTypedColumnPtr queried_aggregation_results_typed =
        boost::dynamic_pointer_cast<OIDTypedColumnType>(
            aggregation_result.second);
    assert(queried_aggregation_results_typed != NULL);
    if (queried_aggregation_results_typed->getColumnType() !=
        PLAIN_MATERIALIZED) {
      queried_aggregation_results =
          queried_aggregation_results_typed->copyIntoDenseValueColumn(
              processorSpecification);
    } else {
      queried_aggregation_results =
          boost::dynamic_pointer_cast<Column<TID> >(aggregation_result.second);
      assert(queried_aggregation_results != NULL);
    }

    // get array access
    TID *contig_ids_ = queried_contig_ids->data();
    string *contig_names = queried_contig_names->data();
    TID *rb_ids = queried_aggregation_results->data();

    for (TID i = 0; i < queried_contig_names->size(); i++) {
      contigName2IdMap.insert(
          make_pair(contig_names[i], contig_ids(contig_ids_[i], rb_ids[i])));
    }
  }

  /* structure to store data of one alignment */
  bam1_t *loop_alignment = bam_init1();
  boost::shared_ptr<AlignmentFile> alignmentFile;
  if (_endsWithFoo(path_to_sam_bam_file.c_str(), ".sam") == 0) {
    alignmentFile =
        boost::shared_ptr<AlignmentFile>(new SAMFile(path_to_sam_bam_file));
  } else if (_endsWithFoo(path_to_sam_bam_file.c_str(), ".bam") == 0) {
    alignmentFile =
        boost::shared_ptr<AlignmentFile>(new BAMFile(path_to_sam_bam_file));
  } else {
    out << "ERROR: No bam or sam file specified!" << endl;
    return false;
  }

  end_preparation = getTimestamp() - start;
  start = getTimestamp();

  /* To get a simple indicator how much data is already imported, we
* first read through the complete file and count the alignment */
  /* open BAM file in read mode */
  uint64_t overallCount = 0;
  if (this->verbose) {
    alignmentFile->openFile();
    while (alignmentFile->getNextAlignment(loop_alignment) >= 0) {
      overallCount++;
    }
    if (overallCount == 0) {
      out << "No alignments found in file " << path_to_sam_bam_file
          << "! Aborting ..." << endl;
      return false;
    }
    alignmentFile->closeFile();
  }
  end_counting = getTimestamp() - start;
  start = getTimestamp();

  std::map<TID, cigar_conversion> cigar_conversion_cache;

  // TODO use stepping according overallCount
  uint64_t printProgressStep = ceil(0.001 * overallCount);
  uint64_t printProgressCounter = 0;
  /* After counting scan again and use the overall count to determine the
   * progress */
  uint64_t count = 0;

  // map to store mate reads
  map<string, bam1_t *> mate_map;

  // measure time for import
  start = getTimestamp();
  // open file again and start actual import
  alignmentFile->openFile();

  string query_template_name;

  int32_t chromosome_id;
  int32_t chromosome_id_mate;

  int32_t position;
  int32_t position_mate;

  string own_key = "";
  string mate_key = "";

  uint64_t cigar_cache_hits = 0;

  string contigName = "";
  string contigNameMate = "";
  TID id_of_first_reference_base_in_contig = UINT64MAX;
  TID id_of_first_reference_base_in_contig_mate = UINT64MAX;
  // TID contig_id;
  bam1_core_t core;
  if (this->verbose) {
    _drawProgress(overallCount, count, 50, out);
  }
  while (alignmentFile->getNextAlignment(loop_alignment) >= 0) {
    bam1_t *alignment = bam_init1();
    bam_copy1(alignment, loop_alignment);
    core = alignment->core;
    // if ((core.flag & BAM_FUNMAP) == 0) {
    query_template_name = bam1_qname(alignment);  // QNAME
    chromosome_id = core.tid;                     // RNAME
    chromosome_id_mate = core.mtid;               // RNEXT
    position = core.pos;                          // POS
    position_mate = core.mpos;                    // RPOS
    // TODO one of these is faster:
    // http:/("");/stackoverflow.com/questions/191757/c-concatenate-string-and-int
    std::stringstream sstm;
    sstm << query_template_name << chromosome_id << position;
    own_key = sstm.str();
    // reset string stream
    sstm.str("");
    sstm << query_template_name << chromosome_id_mate << position_mate;
    mate_key = sstm.str();

    map<string, bam1_t *>::const_iterator mate_iterator =
        mate_map.find(mate_key);
    if (mate_iterator != mate_map.end()) {
      // mate found - import both
      // check whether mate can be imported
      // An offset below 0 means that the alignment has no coordinates, we do
      // not support this in the
      // moment. Moreover, the offset is used for looking up contig IDs. If the
      // offset is greater than the
      // array holding the contig ids, we get undeterministic behavior.
      bam1_t *alignment_mate = mate_iterator->second;
      bam1_core_t mate_core = alignment_mate->core;
      TID import_mate_id = UINT64MAX;
      if ((mate_core.flag & BAM_FUNMAP) == 0 && mate_core.tid >= 0 &&
          mate_core.tid < alignmentFile->getHeader()->n_targets) {
        contigNameMate =
            string(alignmentFile->getHeader()->target_name[chromosome_id_mate]);
        map<string, contig_ids>::iterator it =
            contigName2IdMap.find(contigNameMate);
        if (it != contigName2IdMap.end()) {
          id_of_first_reference_base_in_contig_mate =
              it->second.id_of_first_reference_base;
        } else {
          //    out << "Could not find contig id in database for contig name "
          //    << contigName << ". Skipping alignment ..." << endl;
          id_of_first_reference_base_in_contig_mate = UINT64MAX;
        }
        //}
        import_mate_id = read_id;
        read_id++;
      }
      TID import_read_id = UINT64MAX;
      if ((core.flag & BAM_FUNMAP) == 0 && chromosome_id >= 0 &&
          chromosome_id < alignmentFile->getHeader()->n_targets) {
        contigName =
            string(alignmentFile->getHeader()->target_name[chromosome_id]);
        map<string, contig_ids>::iterator it =
            contigName2IdMap.find(contigName);
        if (it != contigName2IdMap.end()) {
          id_of_first_reference_base_in_contig =
              it->second.id_of_first_reference_base;
        } else {
          //    out << "Could not find contig id in database for contig name "
          //    << contigName << ". Skipping alignment ..." << endl;
          id_of_first_reference_base_in_contig = UINT64MAX;
        }
        //}
        import_read_id = read_id;
        read_id++;
      }

      if (import_mate_id < UINT64MAX &&
          id_of_first_reference_base_in_contig_mate < UINT64MAX) {
        uint32_t *cigar = bam1_cigar(alignment_mate);
        // TID cigar_lookup_key = 0;//string_key_maker.str();
        // std::stringstream string_key_maker;
        // for (int x = 1; x <= mate_core.n_cigar; x++) {
        //    cigar_lookup_key += x * cigar[x];// << mate_core.n_cigar;// <<
        //    position_mate;
        //}
        // assert(cigar_lookup_key > 0);
        // map<TID, cigar_conversion>::iterator it =
        // cigar_conversion_cache.find(
        //        cigar_lookup_key);
        cigar_conversion converted_cigar(mate_core.l_qseq);
        // if (it == cigar_conversion_cache.end()) {
        /* do CIGAR conversion */
        this->_convertCIGAR(cigar, mate_core.n_cigar, mate_core.pos,
                            &converted_cigar);
        //    cigar_conversion_cache.insert(std::make_pair(cigar_lookup_key,
        //    converted_cigar));
        //} else {
        //    cigar_cache_hits++;
        //    converted_cigar = it->second;
        //}
        _importAlignment(sample_genome_id,
                         id_of_first_reference_base_in_contig_mate,
                         alignment_mate, mate_core, import_mate_id,
                         import_read_id, sample_base_id, converted_cigar);
      } else {
        // no coordinates, unmapped read FIXME
        ++numberOfUnmappedReads;
      }
      if (import_read_id < UINT64MAX &&
          id_of_first_reference_base_in_contig < UINT64MAX) {
        uint32_t *cigar = bam1_cigar(alignment);
        // TID cigar_lookup_key = 0;//string_key_maker.str();
        // std::stringstream string_key_maker;
        // for (int x = 1; x <= core.n_cigar; x++) {
        //    cigar_lookup_key += x * cigar[x];// << core.n_cigar;// <<
        //    position;
        //}
        // assert(cigar_lookup_key > 0);
        // map<TID, cigar_conversion>::iterator it =
        // cigar_conversion_cache.find(
        //        cigar_lookup_key);
        cigar_conversion converted_cigar(core.l_qseq);
        // if (it == cigar_conversion_cache.end()) {
        /* do CIGAR conversion */
        this->_convertCIGAR(cigar, core.n_cigar, core.pos, &converted_cigar);
        //    cigar_conversion_cache.insert(std::make_pair(cigar_lookup_key,
        //    converted_cigar));
        //} else {
        //    cigar_cache_hits++;
        //    converted_cigar = it->second;
        //}
        _importAlignment(sample_genome_id, id_of_first_reference_base_in_contig,
                         alignment, core, import_read_id, import_mate_id,
                         sample_base_id, converted_cigar);
      } else {
        // no coordinates, unmapped read FIXME
        ++numberOfUnmappedReads;
      }
      bam_destroy1(alignment_mate);
      bam_destroy1(alignment);
      mate_map.erase(mate_key);
      count += 2;

      if (this->verbose && count > printProgressCounter) {
        _drawProgress(overallCount, count, 50, out);
        printProgressCounter += printProgressStep;
      }
    } else {
      // mate not found - so this must be the first in the pair - add to map
      // TODO check that key does not exist -> failure!
      mate_map.insert(make_pair(own_key, alignment));
    }
    numberOfReads++;
  }
  bam_destroy1(loop_alignment);
  // iterate through map to insert alignments without mates
  typedef std::map<string, bam1_t *>::iterator mate_map_iterator_type;
  // lastContigName = "";
  for (mate_map_iterator_type read_iterator = mate_map.begin();
       read_iterator != mate_map.end(); read_iterator++) {
    // Repeat if you also want to iterate through the second map.
    bam1_t *alignment_nomate = read_iterator->second;
    bam1_core_t nomate_core = alignment_nomate->core;
    TID import_id = UINT64MAX;
    if ((nomate_core.flag & BAM_FUNMAP) == 0 && nomate_core.tid >= 0 &&
        nomate_core.tid < alignmentFile->getHeader()->n_targets) {
      contigName =
          string(alignmentFile->getHeader()->target_name[nomate_core.tid]);
      map<string, contig_ids>::iterator it = contigName2IdMap.find(contigName);
      if (it != contigName2IdMap.end()) {
        id_of_first_reference_base_in_contig =
            it->second.id_of_first_reference_base;
      } else {
        //    out << "Could not find contig id in database for contig name "
        //    << contigName << ". Skipping alignment ..." << endl;
        id_of_first_reference_base_in_contig = UINT64MAX;
      }
      import_id = read_id;
      read_id++;
    }
    if (import_id < UINT64MAX &&
        id_of_first_reference_base_in_contig < UINT64MAX) {
      uint32_t *cigar = bam1_cigar(alignment_nomate);
      // TID cigar_lookup_key = 0;//string_key_maker.str();
      // std::stringstream string_key_maker;
      // for (int x = 1; x <= core.n_cigar; x++) {
      //    cigar_lookup_key += x * cigar[x];// << core.n_cigar;// << position;
      //}
      // assert(cigar_lookup_key > 0);
      // map<TID, cigar_conversion>::iterator it =
      // cigar_conversion_cache.find(cigar_lookup_key);
      cigar_conversion converted_cigar(nomate_core.l_qseq);
      // if (it == cigar_conversion_cache.end()) {
      /* do CIGAR conversion */
      this->_convertCIGAR(cigar, nomate_core.n_cigar, nomate_core.pos,
                          &converted_cigar);
      //    cigar_conversion_cache.insert(std::make_pair(cigar_lookup_key,
      //    converted_cigar));
      //} else {
      //    cigar_cache_hits++;
      //    converted_cigar = it->second;
      //}
      _importAlignment(sample_genome_id, id_of_first_reference_base_in_contig,
                       alignment_nomate, nomate_core, import_id, -1,
                       sample_base_id, converted_cigar);
    } else {
      ++numberOfUnmappedReads;
    }
    count++;
    if (this->verbose && count > printProgressCounter) {
      _drawProgress(overallCount, count, 50, out);
      printProgressCounter += printProgressStep;
    }
    bam_destroy1(alignment_nomate);
  }
  if (this->verbose) {
    _drawProgress(overallCount, overallCount, 50, out, true);
  }
  alignmentFile->closeFile();

  sampleGenomeTable->setNumberofRows(sample_genome_id + 1);
  readTable->setNumberofRows(read_id);
  sampleBaseTable->setNumberofRows(sample_base_id);
  end_importing = getTimestamp() - start;
  start = getTimestamp();

  // Make it persistent
  storeTable(sampleGenomeTable);
  storeTable(readTable);
  storeTable(sampleBaseTable);
  end_storing = getTimestamp() - start;

  out << "Time statistics:" << endl;
  out << fixed << showpoint << setprecision(2);
  std::pair<double, string> time =
      _getTimeAsHumanReadableString(end_preparation);
  out << "Time to PREPARE import: " << time.first << time.second << " ("
      << end_preparation << "ns)" << endl;
  time = _getTimeAsHumanReadableString(end_counting);
  out << "Time to COUNT alignments: " << time.first << time.second << " ("
      << end_counting << "ns)" << endl;
  time = _getTimeAsHumanReadableString(end_importing);
  out << "Time to IMPORT alignments: " << time.first << time.second << " ("
      << end_importing << "ns)" << endl;
  time = _getTimeAsHumanReadableString(end_storing);
  out << "Time to STORE data on disk: " << time.first << time.second << " ("
      << end_storing << "ns)" << endl;
  out << "CIGAR cache hits: " << cigar_cache_hits << endl;
  return true;
}

bool BaseCentric_SimpleKey_Importer::importSampleGenomeData(
    const string path_to_sam_bam_file, const string reference_genome_name,
    const string sample_genome_name) {
  // importing reference genome data
  this->out << "Importing sample genome data" << endl;
  this->out << "SAM/BAM file: " << path_to_sam_bam_file << endl;
  this->out << "Reference genome name: " << reference_genome_name << endl;
  this->out << "Sample genome name: " << sample_genome_name << endl;

  TID reference_genome_id;
  // Determine current reference genome id
  {
    ProcessorSpecification proc_spec(hype::PD0);
    SelectionParam param(proc_spec, ValueConstantPredicate,
                         reference_genome_name, EQUAL);
    PositionListPtr result = rg_name_column->selection(param);
    if (result->empty()) {
      out << "ERROR: Couldn't find " << reference_genome_name
          << " genome in the database." << endl;
      return false;
    }
    // If found the first TID is our reference genome id
    TID *tid = result->begin();
    reference_genome_id = *tid;
  }

  TID numberOfUnmappedReads = 0;
  TID numberOfReads = 0;
  bool success = false;

  success = _importSampleGenomeData(sample_genome_name, reference_genome_id,
                                    path_to_sam_bam_file, numberOfUnmappedReads,
                                    numberOfReads);
  if (success) {
    out << " SUCCESS!" << endl;
    out << "Following tables were imported:" << endl;
    out << sampleGenomeTable->getName() << "("
        << sampleGenomeTable->getNumberofRows() << " rows, "
        << sampleGenomeTable->getSizeinBytes() << " bytes)" << endl;
    out << readTable->getName() << "(" << readTable->getNumberofRows()
        << " rows, " << readTable->getSizeinBytes() << " bytes)" << endl;
    out << sampleBaseTable->getName() << "("
        << sampleBaseTable->getNumberofRows() << " rows, "
        << sampleBaseTable->getSizeinBytes() << " bytes)" << endl;
    if (numberOfReads > 0) {
      out << "Sample genome statistics:" << endl;
      out << "Overall reads: " << numberOfReads << endl;
      out << "Number of Unmapped reads: " << numberOfUnmappedReads << " ( "
          << setprecision(2)
          << (double(numberOfUnmappedReads) / numberOfReads * 100) << "% )"
          << endl;
      out << "Number of Mapped reads: "
          << (numberOfReads - numberOfUnmappedReads) << " ( " << setprecision(2)
          << (double(numberOfReads - numberOfUnmappedReads) / numberOfReads *
              100)
          << "% )" << endl;
    }
  } else {
    out << " ERROR!" << endl;
  }
  return success;
}
}

#endif
