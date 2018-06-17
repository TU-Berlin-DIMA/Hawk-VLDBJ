//
// Created by Sebastian Dorok on 06.07.15.
//

#include <core/global_definitions.hpp>

#ifdef BAM_FOUND

#include <samtools/bam.h>
#include <samtools/bam.h>
#include <core/global_definitions.hpp>
#include <core/variable_manager.hpp>
#include <util/genomics_extension/sequence_centric_simple_key_importer.hpp>

namespace CoGaDB {

// taken from stdint.h
#define UINT64MAX 18446744073709551615UL

using namespace std;

bool SequenceCentric_SimpleKey_Importer::_createDatabase() {
  out << "Creating genome database" << endl;
  out << "Type: sequence centric with simple keys" << endl;
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
    c_seq_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(C_SEQ_COL_NAME, VARCHAR));
    c_rg_id_column = OIDTypedColumnPtr(
        new RLECompressedColumn<TID>(C_REF_GENOME_ID_COL_NAME, OID));

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
    r_flag_column = IntTypedColumnPtr(
        new BitPackedDictionaryCompressedColumn<int>(R_FLAG_COL_NAME, INT, 16));
    r_c_id_column =
        OIDTypedColumnPtr(new RLECompressedColumn<TID>(R_C_ID_COL_NAME, OID));
    r_pos_column = IntTypedColumnPtr(new Column<int>(R_POS_COL_NAME, INT));
    r_map_qual_column = IntTypedColumnPtr(
        new RLECompressedColumn<int>(R_MAPPING_QUALITY_COL_NAME, INT));
    r_cigar_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(R_CIGAR_COL_NAME, VARCHAR));
    r_seq_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(R_SEQ_COL_NAME, VARCHAR));
    r_qual_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(R_QUAL_COL_NAME, VARCHAR));

    r_sg_id_column = OIDTypedColumnPtr(
        new RLECompressedColumn<TID>(R_SAMPLE_GENOME_ID_COL_NAME, OID));
    // FIXME mate information - no compression yet
    r_mate_id_column =
        OIDTypedColumnPtr(new Column<TID>(R_MATE_ID_COL_NAME, OID));

    read_aux_id_column = OIDTypedColumnPtr(
        new VoidCompressedColumnNumber<TID>(RAUX_ID_COL_NAME, OID));
    read_aux_read_id_column = OIDTypedColumnPtr(
        new RLECompressedColumn<TID>(RAUX_READ_ID_COL_NAME, OID));
    // 128 unique tags
    read_aux_tag_column = StringTypedColumnPtr(
        new BitPackedDictionaryCompressedColumn<std::string>(RAUX_TAG_COL_NAME,
                                                             VARCHAR, 8));
    // 16 possible types
    read_aux_type_column = StringTypedColumnPtr(
        new BitPackedDictionaryCompressedColumn<std::string>(RAUX_TYPE_COL_NAME,
                                                             VARCHAR, 4));
    read_aux_value_column =
        StringTypedColumnPtr(new DictionaryCompressedColumn<std::string>(
            RAUX_VALUE_COL_NAME, VARCHAR));
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
    c_seq_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(C_SEQ_COL_NAME, VARCHAR));
    c_rg_id_column =
        OIDTypedColumnPtr(new Column<TID>(C_REF_GENOME_ID_COL_NAME, OID));

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
    r_flag_column = IntTypedColumnPtr(new Column<int>(R_FLAG_COL_NAME, INT));
    r_c_id_column = OIDTypedColumnPtr(new Column<TID>(R_C_ID_COL_NAME, OID));
    r_pos_column = IntTypedColumnPtr(new Column<int>(R_POS_COL_NAME, INT));
    r_map_qual_column =
        IntTypedColumnPtr(new Column<int>(R_MAPPING_QUALITY_COL_NAME, INT));
    r_cigar_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(R_CIGAR_COL_NAME, VARCHAR));
    r_seq_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(R_SEQ_COL_NAME, VARCHAR));
    r_qual_column = StringTypedColumnPtr(
        new DictionaryCompressedColumn<std::string>(R_QUAL_COL_NAME, VARCHAR));

    r_sg_id_column =
        OIDTypedColumnPtr(new Column<TID>(R_SAMPLE_GENOME_ID_COL_NAME, OID));
    // FIXME mate information - no compression yet
    r_mate_id_column =
        OIDTypedColumnPtr(new Column<TID>(R_MATE_ID_COL_NAME, OID));

    read_aux_id_column =
        OIDTypedColumnPtr(new Column<TID>(RAUX_ID_COL_NAME, OID));
    read_aux_read_id_column =
        OIDTypedColumnPtr(new Column<TID>(RAUX_READ_ID_COL_NAME, OID));
    read_aux_tag_column =
        StringTypedColumnPtr(new DictionaryCompressedColumn<std::string>(
            RAUX_TAG_COL_NAME, VARCHAR));
    read_aux_type_column =
        StringTypedColumnPtr(new DictionaryCompressedColumn<std::string>(
            RAUX_TYPE_COL_NAME, VARCHAR));
    read_aux_value_column =
        StringTypedColumnPtr(new DictionaryCompressedColumn<std::string>(
            RAUX_VALUE_COL_NAME, VARCHAR));
  }

  // Setting up tables inclusive primary and foreign keys
  vector<ColumnPtr> ref_genome_cols;
  vector<ColumnPtr> contig_cols;
  vector<ColumnPtr> ref_base_cols;
  vector<ColumnPtr> sample_genome_cols;
  vector<ColumnPtr> read_cols;
  vector<ColumnPtr> read_aux_cols;
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
  contig_cols.push_back(c_seq_column);
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
  read_cols.push_back(r_flag_column);
  read_cols.push_back(r_c_id_column);
  read_cols.push_back(r_pos_column);
  read_cols.push_back(r_map_qual_column);
  read_cols.push_back(r_cigar_column);
  read_cols.push_back(r_seq_column);
  read_cols.push_back(r_qual_column);
  read_cols.push_back(r_sg_id_column);
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
  if (!readTable->setForeignKeyConstraint(R_C_ID_COL_NAME, C_ID_COL_NAME,
                                          C_TBL_NAME)) {
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

  // read aux
  out << "Creating " << RAUX_TBL_NAME << " table ..." << endl;
  read_aux_cols.push_back(read_aux_id_column);
  read_aux_cols.push_back(read_aux_read_id_column);
  read_aux_cols.push_back(read_aux_tag_column);
  read_aux_cols.push_back(read_aux_type_column);
  read_aux_cols.push_back(read_aux_value_column);
  readAuxTable = TablePtr(new Table(RAUX_TBL_NAME, read_aux_cols));
  if (!readAuxTable->setPrimaryKeyConstraint(RAUX_ID_COL_NAME)) {
    COGADB_ERROR("Failed to set Primary Key Constraint!", "");
    return false;
  }
  if (!readAuxTable->setForeignKeyConstraint(RAUX_READ_ID_COL_NAME,
                                             R_ID_COL_NAME, R_TBL_NAME)) {
    COGADB_ERROR("Failed to set Foreign Key Constraint!", "");
    return false;
  }
  addToGlobalTableList(readAuxTable);

  // Store the tables on disk
  storeTable(refGenomeTable);
  storeTable(contigTable);
  storeTable(sampleGenomeTable);
  storeTable(readTable);
  storeTable(readAuxTable);
  return true;
}

bool SequenceCentric_SimpleKey_Importer::_initDatabaseConnection() {
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
    if ((*table)->getName().compare(RAUX_TBL_NAME) == 0) {
      readAuxTable = *table;
      existing_tables++;
      continue;
    }
  }
  if (existing_tables == 5) {
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
    r_flag_column = boost::dynamic_pointer_cast<IntTypedColumnType>(
        readTable->getColumnbyName(R_FLAG_COL_NAME));
    assert(r_flag_column != NULL);
    r_c_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        readTable->getColumnbyName(R_C_ID_COL_NAME));
    assert(r_flag_column != NULL);
    r_pos_column = boost::dynamic_pointer_cast<IntTypedColumnType>(
        readTable->getColumnbyName(R_POS_COL_NAME));
    assert(r_flag_column != NULL);
    r_map_qual_column = boost::dynamic_pointer_cast<IntTypedColumnType>(
        readTable->getColumnbyName(R_MAPPING_QUALITY_COL_NAME));
    assert(r_map_qual_column != NULL);
    r_cigar_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        readTable->getColumnbyName(R_CIGAR_COL_NAME));
    assert(r_cigar_column != NULL);
    r_seq_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        readTable->getColumnbyName(R_SEQ_COL_NAME));
    assert(r_seq_column != NULL);
    r_qual_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        readTable->getColumnbyName(R_QUAL_COL_NAME));
    assert(r_qual_column != NULL);
    r_sg_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        readTable->getColumnbyName(R_SAMPLE_GENOME_ID_COL_NAME));
    assert(r_sg_id_column != NULL);
    r_mate_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        readTable->getColumnbyName(R_MATE_ID_COL_NAME));
    assert(r_mate_id_column != NULL);

    read_aux_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        readAuxTable->getColumnbyName(RAUX_ID_COL_NAME));
    assert(read_aux_id_column != NULL);
    read_aux_read_id_column = boost::dynamic_pointer_cast<OIDTypedColumnType>(
        readAuxTable->getColumnbyName(RAUX_READ_ID_COL_NAME));
    assert(read_aux_read_id_column != NULL);
    read_aux_tag_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        readAuxTable->getColumnbyName(RAUX_TAG_COL_NAME));
    assert(read_aux_tag_column != NULL);
    read_aux_type_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        readAuxTable->getColumnbyName(RAUX_TYPE_COL_NAME));
    assert(read_aux_type_column != NULL);
    read_aux_value_column = boost::dynamic_pointer_cast<StringTypedColumnType>(
        readAuxTable->getColumnbyName(RAUX_VALUE_COL_NAME));
    assert(read_aux_value_column != NULL);
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

SequenceCentric_SimpleKey_Importer::SequenceCentric_SimpleKey_Importer(
    ostream &_out, bool _compress_data, bool _verbose)
    : GenomeDataImporter(_out, _compress_data, _verbose) {
  this->_initDatabaseConnection();
}

bool SequenceCentric_SimpleKey_Importer::_importReferenceGenomeData(
    const string reference_genome_name, const string path_to_fasta_file) {
  Timestamp start = getTimestamp();
  Timestamp end_preparation = 0;
  Timestamp end_importing = 0;
  Timestamp end_storing = 0;

  // initiate id
  TID reference_genome_id = refGenomeTable->getNumberofRows();
  TID contigId = contigTable->getNumberofRows();

  rg_id_column->insert(reference_genome_id);
  rg_name_column->insert(reference_genome_name);

  // Insert data from FASTA file
  string fastaLine;
  string contig_sequence = "";
  bool first_contig_inserted = false;
  ifstream fastaFile(path_to_fasta_file.c_str());

  // measure file length
  fastaFile.seekg(0, fastaFile.end);
  uint64_t lengthOverall = static_cast<uint64_t>(fastaFile.tellg());
  uint64_t printProgressStep =
      static_cast<uint64_t>(ceil(0.001 * lengthOverall));
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
        if (first_contig_inserted) {
          c_seq_column->insert(contig_sequence);
        }
        c_id_column->insert(contigId);
        c_name_column->insert(fastaLine.substr(1, firstWhiteSpace));
        c_rg_id_column->insert(reference_genome_id);
        contigId++;
        first_contig_inserted = true;
      } else {
        /* bases for reference found */
        contig_sequence.append(fastaLine);
      }
      length = static_cast<uint64_t>(fastaFile.tellg());
      if (this->verbose && length > printProgressCounter) {
        _drawProgress(lengthOverall, length, 50, out);
        printProgressCounter += printProgressStep;
      }
    }
    // insert sequence of last contig
    c_seq_column->insert(contig_sequence);
    if (this->verbose) {
      _drawProgress(lengthOverall, lengthOverall, 50, out, true);
    }
    fastaFile.close();

    refGenomeTable->setNumberofRows(reference_genome_id + 1);
    contigTable->setNumberofRows(contigId);
  } else {
    out << "ERROR: Unable to open file: '" << path_to_fasta_file << "'" << endl;
    return false;
  }

  end_importing = getTimestamp() - start;
  start = getTimestamp();

  // Store data
  storeTable(refGenomeTable);
  storeTable(contigTable);
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

bool SequenceCentric_SimpleKey_Importer::importReferenceGenomeData(
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
  } else {
    out << " ERROR!" << endl;
  }
  return success;
}

// converts auxiliary data into a vector of aux_info structs
std::vector<GenomeDataImporter::aux_info>
SequenceCentric_SimpleKey_Importer::getAuxiliaryStringVector(
    bam1_t *alignment) {
  uint8_t *auxiliary_info = bam1_aux(alignment);
  std::vector<aux_info> result;
  for (int j = 0; j < alignment->l_aux;) {
    uint8_t tag1 = auxiliary_info[j++];
    uint8_t tag2 = auxiliary_info[j++];
    uint8_t type = auxiliary_info[j++];
    string type_string;
    std::stringstream value_string;
    char *str = 0;
    switch (type) {
      case 'C':
      case 'c':
      case 'A':
        value_string << static_cast<char>(auxiliary_info[j]);
        j += 1;
        type_string = "A";
        break;
      case 'S':
        value_string << static_cast<uint16_t>(auxiliary_info[j]);
        j += 1;
        type_string = "S";
        break;
      case 's':
        value_string << static_cast<int16_t>(auxiliary_info[j]);
        type_string = "s";
        j += 2;
        break;
      case 'I':
        value_string << static_cast<uint32_t>(auxiliary_info[j]);
        type_string = "I";
        j += 4;
        break;
      case 'i':
        value_string << static_cast<int32_t>(auxiliary_info[j]);
        type_string = "i";
        j += 4;
        break;
      case 'f':
      case 'F':
        value_string << static_cast<float>(auxiliary_info[j]);
        type_string = "f";
        j += 4;
        break;
      case 'Z':
        str = bam_aux2Z(auxiliary_info + 1);
        value_string << str;
        type_string = "Z";
        j += value_string.str().length() + 1;
        break;
      case 'H':
      // not supported yet
      case 'B':
      // not supported yet
      default:
        continue;
    }

    std::stringstream tag_string;
    tag_string << tag1 << tag2;

    result.push_back(
        aux_info(tag_string.str(), type_string, value_string.str()));
  }
  return result;
}

/**
* bam1_t* alignment ... to convert
* int* numberOfUnmappedReads ... for statistics
* int* numberOfReads ... for statistics
*/
bool SequenceCentric_SimpleKey_Importer::_importAlignment(
    TID sample_genome_id, TID id_of_contig, bam1_t *alignment,
    const bam1_core_t c, TID read_id, TID mate_id, TID &raux_id,
    string converted_cigar) {
  int32_t pos = c.pos;

  uint8_t *baseSequence, *qualityValues;

  /* extract alignment and quality values */
  baseSequence = bam1_seq(alignment);
  qualityValues = bam1_qual(alignment);

  /* read_id, mapping quality, description */
  r_id_column->insert(read_id);
  r_qname_column->insert(string(bam1_qname(alignment)));
  r_flag_column->insert(static_cast<int>(c.flag));
  r_c_id_column->insert(id_of_contig);
  r_pos_column->insert(pos);
  r_map_qual_column->insert(static_cast<int>(c.qual));
  r_cigar_column->insert(converted_cigar);

  std::stringstream seq_string;
  seq_string.str("");
  for (int i = 0; i < c.l_qseq; i++) {
    seq_string << bam_nt16_rev_table[bam1_seqi(baseSequence, i)];
  }
  r_seq_column->insert(seq_string.str());

  std::stringstream qual_string;
  qual_string.str("");
  for (int i = 0; i < c.l_qseq; i++) {
    qual_string << qualityValues[i];
  }
  r_qual_column->insert(qual_string.str());
  r_sg_id_column->insert(sample_genome_id);
  r_mate_id_column->insert(mate_id);

  vector<aux_info> aux_infos = getAuxiliaryStringVector(alignment);
  vector<aux_info>::const_iterator aux_it;
  for (aux_it = aux_infos.begin(); aux_it != aux_infos.end(); aux_it++) {
    read_aux_id_column->insert(raux_id);
    read_aux_read_id_column->insert(read_id);
    read_aux_tag_column->insert(aux_it->tag);
    read_aux_type_column->insert(aux_it->type);
    read_aux_value_column->insert(aux_it->value);
    raux_id++;
  }
  return true;
}

string SequenceCentric_SimpleKey_Importer::convertCIGAR2String(
    uint32_t *cigar, uint32_t cigarLength) {
  std::stringstream cigar_string;
  cigar_string.str("");
  for (uint32_t i = 0; i < cigarLength; ++i) {
    switch (bam_cigar_op(*cigar)) {
      case BAM_CMATCH: /* M = match or mismatch*/
        cigar_string << bam_cigar_oplen(*cigar) << "M";
        break;
      case BAM_CEQUAL: /* equals = match */
        cigar_string << bam_cigar_oplen(*cigar) << "=";
        break;
      case BAM_CDIFF: /* X = mismatch */
        cigar_string << bam_cigar_oplen(*cigar) << "X";
        break;
      case BAM_CINS: /* I = insertion to the reference */
        cigar_string << bam_cigar_oplen(*cigar) << "I";
        break;
      case BAM_CDEL: /* D = deletion from the reference */
        cigar_string << bam_cigar_oplen(*cigar) << "D";
        break;
      case BAM_CSOFT_CLIP:
        /* S = clip on the read with clipped sequence present in qseq */
        cigar_string << bam_cigar_oplen(*cigar) << "S";
        break;
      case BAM_CREF_SKIP: /* N = skip on the reference (e.g. spliced alignment)
                             */
        /* according to the SAM Format reference, this CIGAR operation is only
     * defined
     * for mRNA-to-genome mapping! */
        /* according to
     * http://davetang.org/wiki/tiki-index.php?page=SAM#Spliced_alignment it can
     * be
     * handled like a deletion operator without inserting artificial 'X' bases
     */
        /* TODO just log it */
        cigar_string << bam_cigar_oplen(*cigar) << "N";
        break;
      case BAM_CHARD_CLIP: /* H = clip on the read with clipped sequence trimmed
                              off */
        /* hard clipping means there were bases that do not match the reference
     * but they are not in the
     * sequence anymore -> nothing to do just ignore this */
        /* TODO just log it */
        cigar_string << bam_cigar_oplen(*cigar) << "H";
        break;
      case BAM_CPAD: /* P = padding */
        /* we can ignore the padding operation as it tells us that there is an
     * deletion in the read and
     * also the reference in accordance to another read that inserts bases at
     * this position,
     * see http://davetang.org/wiki/tiki-index.php?page=SAM#Padded_alignment */
        /* TODO just log it */
        cigar_string << bam_cigar_oplen(*cigar) << "P";
        break;
        /* BAM_CBACK is not available in all samtools versions */
        /* case BAM_CBACK: */
        /* currently not in the format reference but in the source code */
        /* TODO just log it */
        /* break; */
    }
    cigar += 1;
  }
  return cigar_string.str();
}

bool SequenceCentric_SimpleKey_Importer::_importSampleGenomeData(
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
  TID raux_id = readAuxTable->getNumberofRows();

  sg_id_column->insert(sample_genome_id);
  sg_name_column->insert(sample_genome_name);

  // Prepare import information: Contig names and their database id
  // The corresponding SQL query: select c_name, c_id from contig
  // where c_rg_id = <reference_genome_id>;
  map<string, TID> contigName2IdMap;
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

    boost::shared_ptr<Column<TID> > queried_contig_ids;
    boost::shared_ptr<Column<string> > queried_contig_names;

    OIDTypedColumnPtr selected_contig_ids_typed =
        boost::dynamic_pointer_cast<OIDTypedColumnType>(selected_contig_ids);
    assert(selected_contig_ids_typed != NULL);
    if (selected_contig_ids_typed->getColumnType() != PLAIN_MATERIALIZED) {
      queried_contig_ids = selected_contig_ids_typed->copyIntoDenseValueColumn(
          processorSpecification);
    } else {
      queried_contig_ids =
          boost::dynamic_pointer_cast<Column<TID> >(selected_contig_ids);
      assert(queried_contig_ids != NULL);
    }

    StringTypedColumnPtr selected_contig_names_typed =
        boost::dynamic_pointer_cast<StringTypedColumnType>(
            selected_contig_names);
    assert(selected_contig_names_typed != NULL);
    queried_contig_names =
        selected_contig_names_typed->copyIntoDenseValueColumn(
            processorSpecification);

    TID *contig_ids_ = queried_contig_ids->data();
    string *contig_names = queried_contig_names->data();

    for (TID i = 0; i < queried_contig_ids->size(); i++) {
      contigName2IdMap.insert(make_pair(contig_names[i], contig_ids_[i]));
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
  uint64_t printProgressStep =
      static_cast<uint64_t>(ceil(0.001 * overallCount));
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
  TID id_of_contig = UINT64MAX;
  TID id_of_contig_mate = UINT64MAX;

  bam1_core_t core;
  if (this->verbose) {
    _drawProgress(overallCount, count, 50, out);
  }
  while (alignmentFile->getNextAlignment(loop_alignment) >= 0) {
    bam1_t *alignment = bam_init1();
    bam_copy1(alignment, loop_alignment);
    core = alignment->core;

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
        map<string, TID>::iterator it = contigName2IdMap.find(contigNameMate);
        if (it != contigName2IdMap.end()) {
          id_of_contig_mate = it->second;
        } else {
          //    out << "Could not find contig id in database for contig name "
          //    << contigName << ". Skipping alignment ..." << endl;
          id_of_contig_mate = UINT64MAX;
        }
        import_mate_id = read_id;
        read_id++;
      }
      TID import_read_id = UINT64MAX;
      if ((core.flag & BAM_FUNMAP) == 0 && chromosome_id >= 0 &&
          chromosome_id < alignmentFile->getHeader()->n_targets) {
        contigName =
            string(alignmentFile->getHeader()->target_name[chromosome_id]);
        map<string, TID>::iterator it = contigName2IdMap.find(contigName);
        if (it != contigName2IdMap.end()) {
          id_of_contig = it->second;
        } else {
          //    out << "Could not find contig id in database for contig name "
          //    << contigName << ". Skipping alignment ..." << endl;
          id_of_contig = UINT64MAX;
        }
        //}
        import_read_id = read_id;
        read_id++;
      }

      if (import_mate_id < UINT64MAX && id_of_contig_mate < UINT64MAX) {
        uint32_t *cigar = bam1_cigar(alignment_mate);
        /* do CIGAR conversion */
        string converted_cigar = convertCIGAR2String(cigar, mate_core.n_cigar);
        _importAlignment(sample_genome_id, id_of_contig_mate, alignment_mate,
                         mate_core, import_mate_id, import_read_id, raux_id,
                         converted_cigar);
      } else {
        // no coordinates, unmapped read FIXME
        ++numberOfUnmappedReads;
      }
      if (import_read_id < UINT64MAX && id_of_contig < UINT64MAX) {
        uint32_t *cigar = bam1_cigar(alignment);
        /* do CIGAR conversion */
        string converted_cigar = convertCIGAR2String(cigar, core.n_cigar);
        _importAlignment(sample_genome_id, id_of_contig, alignment, core,
                         import_read_id, import_mate_id, raux_id,
                         converted_cigar);
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
      map<string, TID>::iterator it = contigName2IdMap.find(contigName);
      if (it != contigName2IdMap.end()) {
        id_of_contig = it->second;
      } else {
        //    out << "Could not find contig id in database for contig name "
        //    << contigName << ". Skipping alignment ..." << endl;
        id_of_contig = UINT64MAX;
      }
      import_id = read_id;
      read_id++;
    }
    if (import_id < UINT64MAX && id_of_contig < UINT64MAX) {
      uint32_t *cigar = bam1_cigar(alignment_nomate);
      /* do CIGAR conversion */
      string converted_cigar = convertCIGAR2String(cigar, nomate_core.n_cigar);
      _importAlignment(sample_genome_id, id_of_contig, alignment_nomate,
                       nomate_core, import_id, -1, raux_id, converted_cigar);
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
  readAuxTable->setNumberofRows(raux_id);
  end_importing = getTimestamp() - start;
  start = getTimestamp();

  // Make it persistent
  storeTable(sampleGenomeTable);
  storeTable(readTable);
  storeTable(readAuxTable);
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

bool SequenceCentric_SimpleKey_Importer::importSampleGenomeData(
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
