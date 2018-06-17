
#include <iostream>
#include <persistence/storage_manager.hpp>
#include <util/result_output_format.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/find_iterator.hpp>
#include <boost/lexical_cast.hpp>
#include <cstdlib>
#include <string>
#include "core/variable_manager.hpp"
#include "util/genomics_extension/CigarFSM.hpp"

namespace CoGaDB {

using namespace std;

class TableFormat : public ResultFormat {
 public:
  static ResultFormatPtr create();
  virtual const std::string getHeader(
      const std::string& table_name, const std::vector<ColumnPtr>& columns,
      const std::vector<unsigned int>& max_column_widths);
  virtual const std::string getRows(
      const std::vector<std::vector<std::string> >& string_columns,
      const std::vector<unsigned int>& max_column_widths);
  virtual const std::string getFooter(
      const std::vector<ColumnPtr>& columns,
      const std::vector<unsigned int>& max_column_widths);
};

ResultFormatPtr TableFormat::create() {
  return ResultFormatPtr(new TableFormat());
}

const std::string TableFormat::getHeader(
    const std::string& table_name, const std::vector<ColumnPtr>& columns_,
    const std::vector<unsigned int>& max_column_widths) {
  std::stringstream ss;
  std::ostream& out = ss;
  std::string name_ = table_name;
  // print string columns
  if (!name_.empty()) out << name_ << ":" << endl;
  out << "+-";
  for (unsigned int i = 0; i < columns_.size(); i++) {
    out << std::string(max_column_widths[i], '-');
    if (i < columns_.size() - 1) out << "-+-";
  }
  out << "-+" << std::endl;
  out << "| ";
  for (unsigned int i = 0; i < columns_.size(); i++) {
    std::string plain_col_name = columns_[i]->getName();
    std::string table_name;
    uint32_t version;
    if (!VariableManager::instance().getVariableValueBoolean(
            "show_versions_in_output")) {
      parseColumnIndentifierName(columns_[i]->getName(), table_name,
                                 plain_col_name, version);
    }
    out << plain_col_name
        << std::string(max_column_widths[i] - plain_col_name.size(), ' ')
        << " | ";
  }
  out << endl;
  out << "+=";
  for (unsigned int i = 0; i < columns_.size(); i++) {
    out << std::string(max_column_widths[i], '=');
    if (i < columns_.size() - 1) out << "=+=";
  }
  out << "=+" << std::endl;
  return ss.str();
}

const std::string TableFormat::getRows(
    const std::vector<std::vector<std::string> >& string_columns,
    const std::vector<unsigned int>& max_column_widths) {
  std::stringstream ss;
  std::ostream& out = ss;
  for (unsigned int j = 0; j < string_columns[0].size(); j++) {
    out << "| ";
    for (unsigned int i = 0; i < string_columns.size(); i++) {
      out << std::string(max_column_widths[i] - string_columns[i][j].size(),
                         ' ')
          << string_columns[i][j] << " | ";
    }
    out << std::endl;
  }
  return ss.str();
}

const std::string TableFormat::getFooter(
    const std::vector<ColumnPtr>& columns_,
    const std::vector<unsigned int>& max_column_widths) {
  std::stringstream ss;
  std::ostream& out = ss;
  out << "+-";
  for (unsigned int i = 0; i < columns_.size(); i++) {
    out << std::string(max_column_widths[i], '-');
    if (i < columns_.size() - 1) out << "-+-";
  }
  out << "-+" << std::endl;
  out << columns_[0]->size() << " rows" << endl;
  return ss.str();
}

class CSVFormat : public ResultFormat {
 public:
  static ResultFormatPtr create();
  virtual const std::string getHeader(
      const std::string& table_name, const std::vector<ColumnPtr>& columns,
      const std::vector<unsigned int>& max_column_widths);
  virtual const std::string getRows(
      const std::vector<std::vector<std::string> >& string_columns,
      const std::vector<unsigned int>& max_column_widths);
  virtual const std::string getFooter(
      const std::vector<ColumnPtr>& columns,
      const std::vector<unsigned int>& max_column_widths);
};

ResultFormatPtr CSVFormat::create() { return ResultFormatPtr(new CSVFormat()); }

const std::string CSVFormat::getHeader(
    const std::string& table_name, const std::vector<ColumnPtr>& columns_,
    const std::vector<unsigned int>& max_column_widths) {
  std::stringstream ss;
  std::ostream& out = ss;
  std::string name_ = table_name;
  for (unsigned int i = 0; i < columns_.size(); i++) {
    std::string plain_col_name = columns_[i]->getName();
    std::string table_name;
    uint32_t version;
    if (!VariableManager::instance().getVariableValueBoolean(
            "show_versions_in_output")) {
      parseColumnIndentifierName(columns_[i]->getName(), table_name,
                                 plain_col_name, version);
    }
    out << plain_col_name
        << std::string(max_column_widths[i] - plain_col_name.size(), ' ');
    if (i + 1 < columns_.size()) out << " | ";
  }
  out << endl;
  return ss.str();
}

const std::string CSVFormat::getRows(
    const std::vector<std::vector<std::string> >& string_columns,
    const std::vector<unsigned int>& max_column_widths) {
  std::stringstream ss;
  std::ostream& out = ss;
  for (unsigned int j = 0; j < string_columns[0].size(); j++) {
    for (unsigned int i = 0; i < string_columns.size(); i++) {
      //                    out <<
      //                    std::string(max_column_widths[i]-string_columns[i][j].size(),'
      //                    ')  << string_columns[i][j];
      out << string_columns[i][j];
      if (i + 1 < string_columns.size()) out << " | ";
    }
    out << std::endl;
  }
  return ss.str();
}

const std::string CSVFormat::getFooter(
    const std::vector<ColumnPtr>& columns_,
    const std::vector<unsigned int>& max_column_widths) {
  return std::string();
}

/*
 * \brief Sam-Formatter
 * Formats Output in Sequence Alignment/Map Format. See SAM spezification for
 * more details:
 * https://samtools.github.io/hts-specs/SAMv1.pdf
 * Only works with Tables created by Queries from
 * util/genomic_extension/samQueryCreator.hpp
 */
class SAMFormat : public ResultFormat {
 public:
  static ResultFormatPtr create();
  virtual const std::string getHeader(
      const std::string& table_name, const std::vector<ColumnPtr>& columns,
      const std::vector<unsigned int>& max_column_widths);
  virtual const std::string getRows(
      const std::vector<std::vector<std::string> >& string_columns,
      const std::vector<unsigned int>& max_column_widths);
  virtual const std::string getFooter(
      const std::vector<ColumnPtr>& columns,
      const std::vector<unsigned int>& max_column_widths);
};

ResultFormatPtr SAMFormat::create() { return ResultFormatPtr(new SAMFormat()); }

const std::string SAMFormat::getHeader(
    const std::string& table_name, const std::vector<ColumnPtr>& columns_,
    const std::vector<unsigned int>& max_column_widths) {
  std::stringstream ss;
  std::ostream& out = ss;

  // Default HD-Line
  out << "@HD\t"
      << "VN:1.0\t"
      << "SO:coordinate" << endl;
  return ss.str();
}

const std::string SAMFormat::getRows(
    const std::vector<std::vector<std::string> >& string_columns,
    const std::vector<unsigned int>& max_column_widths) {
  std::stringstream ss;
  std::ostream& out = ss;

  // Initialisation
  int currentReadId = -1;
  std::string currentQName = "";
  std::string currentFlag = "";
  std::string currentCname = "";
  unsigned int currentSmallestRbPosition = 0;
  std::string currentMapQ = "";
  std::string currentBaseValueSequence = "";
  std::string currentCigarString = "";
  std::string currentQual = "";

  std::string currentSamLine = "";

  std::stringstream headerStringStream;
  std::ostream& headerOutStream = headerStringStream;

  CigarFSM cigarFSM;

  // Row Iteration
  for (unsigned int i = 0; i < string_columns[0].size(); i++) {
    int tmpReadId;
    try {
      tmpReadId =
          boost::lexical_cast<int>(string_columns[0][i]);  // maybe [0][i]]
    } catch (const boost::bad_lexical_cast&) {
      return std::string("DEBUG: Read-ID could not be cast to Integer");
    }

    // Check if Id is new -> initialize new read
    if (tmpReadId != currentReadId) {
      // New Headerline if new contigName. ContigName cannot change if ReadId
      // does not change
      std::string tmpCName = string_columns[3][i];
      if (tmpCName != currentCname) {
        // currentCname = string_columns[3][i]; DEBUG:
        // TODO uncomment if name is not with whitespaces and client is there
        //                                std::string headerLengthQuery =
        //                                "select count(rb_id) as length from "
        //                                        "reference_base join contig on
        //                                        rb_c_id=c_id where c_name=";
        //                                headerLengthQuery+=currentCname;
        //                                headerLengthQuery+=";";
        //                                TablePtr referenceBaseCount =
        //                                CoGaDB::SQL::executeSQL(headerLengthQuery,
        //                                client);
        //                                std::string currentHeaderLength =
        //                                referenceBaseCount->getColumnbyName("length")->getStringValue(0);
        headerOutStream << "@SQ\t"
                        << "SN:" << tmpCName << "\t"
                        << endl;  //"LN:" << currentHeaderLength << endl;
      }
      if (currentReadId != -1) {
        // FIXME put directly in samline calculation
        currentCigarString = cigarFSM.getCigar();

        currentSamLine =
            currentQName + "\t" + currentFlag + "\t" + currentCname + "\t" +
            boost::lexical_cast<string>(currentSmallestRbPosition + 1) + "\t" +
            currentMapQ + "\t" + currentCigarString + "\t" + "*" + "\t" + "0" +
            "\t" + "0" + "\t" + currentBaseValueSequence + "\t" + currentQual;
        out << currentSamLine << endl;

        cigarFSM.reset();
        currentBaseValueSequence = "";
        currentQual = "";
      }

      currentReadId = tmpReadId;
      currentQName = string_columns[1][i];
      currentFlag = string_columns[2][i];
      currentCname = string_columns[3][i];
      try {
        currentSmallestRbPosition =
            boost::lexical_cast<int>(string_columns[4][i]);
      } catch (const boost::bad_lexical_cast&) {
        return std::string("DEBUG: RB-Position could not be cast to Integer");
      }
      currentMapQ = string_columns[5][i];
    }
    // Normal iteration skipps read-initialisation above

    std::string tmpBaseValue = string_columns[6][i];

    cigarFSM.handleDatabaseInput(
        tmpBaseValue, boost::lexical_cast<int>(string_columns[7][i]));
    if (tmpBaseValue != "X") {
      currentBaseValueSequence += tmpBaseValue;
      char tmpMapQ;
      try {
        int tmp = boost::lexical_cast<int>(string_columns[8][i]);
        tmp += 33;
        tmpMapQ = tmp;
      } catch (const boost::bad_lexical_cast&) {
        return std::string(
            "DEBUG: SB_Call_Quality could not be cast to Integer");
      }

      currentQual += tmpMapQ;
    }
  }
  currentCigarString = cigarFSM.getCigar();
  // Write last line FIXME - Better way please.. put one output at the end of
  // the loop
  out << currentQName + "\t" + currentFlag + "\t" + currentCname + "\t" +
             boost::lexical_cast<string>(currentSmallestRbPosition + 1) + "\t" +
             currentMapQ + "\t" + currentCigarString + "\t" + "*" + "\t" + "0" +
             "\t" + "0" + "\t" + currentBaseValueSequence + "\t" + currentQual;
  std::string result = headerStringStream.str() + ss.str();
  return result;
}

const std::string SAMFormat::getFooter(
    const std::vector<ColumnPtr>& columns_,
    const std::vector<unsigned int>& max_column_widths) {
  return std::string();
}

const std::string ResultFormat::getResult(const std::string& table_name,
                                          const std::vector<ColumnPtr>& columns,
                                          bool include_header) {
  std::stringstream ss;
  std::ostream& out = ss;
  std::string name_ = table_name;

  std::vector<ColumnPtr> columns_(columns);

  loadColumnsInMainMemory(columns_);
  if (columns_.empty()) {
    out << "Table " << name_ << " is Empty, nothing to print" << endl;
    return ss.str();
  }

  std::vector<std::vector<std::string> > string_columns(columns_.size());
  std::vector<unsigned int> max_column_widths(columns_.size());
  // init column widths with length of column Names
  for (unsigned int i = 0; i < columns_.size(); i++) {
    std::string plain_col_name = columns_[i]->getName();
    std::string table_name;
    uint32_t version;
    if (!VariableManager::instance().getVariableValueBoolean(
            "show_versions_in_output")) {
      parseColumnIndentifierName(columns_[i]->getName(), table_name,
                                 plain_col_name, version);
    }

    max_column_widths[i] = plain_col_name.size();
    // check whether they are dormant in CPU memory
    if (columns_[i]->getMemoryID() != hype::PD_Memory_0) {
      ColumnPtr tmp = copy(columns_[i], hype::PD_Memory_0);
      assert(tmp != NULL);
      columns_[i] = tmp;
      std::cout << "Copy back to CPU: " << columns_[i]->getName() << std::endl;
    }
  }

  // build string columns
  for (TID j = 0; j < columns_[0]->size(); j++) {
    for (unsigned int i = 0; i < columns_.size(); i++) {
      std::string val = columns_[i]->getStringValue(j);
      if (val.size() > max_column_widths[i]) {
        max_column_widths[i] = val.size();
      }
      string_columns[i].push_back(val);
    }
  }
  unsigned int table_width =
      std::accumulate(max_column_widths.begin(), max_column_widths.end(), 0);
  table_width += 4;  // opening and closing '|' character
  table_width +=
      (columns_.size() - 1) * 3;  // limiting string between columns ' | '
  if (include_header) out << getHeader(table_name, columns, max_column_widths);
  out << getRows(string_columns, max_column_widths);
  out << getFooter(columns, max_column_widths);

  return ss.str();
}

ResultFormatPtr getResultFormatter(const std::string& formatter_name) {
  typedef ResultFormatPtr (*ResultFormatFactoryFunc)();
  typedef std::map<std::string, ResultFormatFactoryFunc> ResultFormatFactoryMap;

  ResultFormatFactoryMap map;
  map.insert(std::make_pair("table", &TableFormat::create));
  map.insert(std::make_pair("csv", &CSVFormat::create));
  map.insert(std::make_pair("sam", &SAMFormat::create));

  ResultFormatFactoryMap::iterator it = map.find(formatter_name);
  if (it != map.end()) {
    // ResultFormatPtr formatter(new TableFormat());
    return it->second();
  } else {
    COGADB_FATAL_ERROR("Invalid Output Format: " << formatter_name, "");
    return ResultFormatPtr();
  }
}
}  // end namespace CoGaDB
