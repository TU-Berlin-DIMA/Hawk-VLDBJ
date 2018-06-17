
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <boost/range/algorithm/count.hpp>
#include <core/attribute_reference.hpp>
#include <core/hash_table.hpp>
#include <core/table.hpp>

#include <core/data_dictionary.hpp>
#include <core/foreign_key_constraint.hpp>
#include <query_compilation/minimal_api.hpp>
#include <util/getname.hpp>

#include <persistence/storage_manager.hpp>
#include <util/dictionary_compressed_col.hpp>
#include <util/functions.hpp>

namespace CoGaDB {

std::pair<bool, AttributeType> getAttributeType(
    const TablePtr table, const std::string& column_name) {
  TableSchema::const_iterator cit;
  TableSchema schema = table->getSchema();
  for (cit = schema.begin(); cit != schema.end(); ++cit) {
    if (compareAttributeReferenceNames(cit->second, column_name)) {
      return std::make_pair(true, cit->first);
    }
  }
  return std::make_pair(false, INT);
}

AttributeReference::AttributeReference(
    const TablePtr _table, const std::string& _input_attribute_name,
    const std::string& _result_attribute_name, uint32_t _version_id)
    : table_ptr_(_table),
      input_attribute_name(_input_attribute_name),
      type_(),
      version_id(_version_id),
      result_attribute_name(_result_attribute_name),
      attr_ref_type(INPUT_ATTRIBUTE) {
  assert(table_ptr_ != NULL && "Invalid TablePtr!");
  std::string fully_qualified_col = input_attribute_name;
  fully_qualified_col += ".";
  fully_qualified_col += boost::lexical_cast<std::string>(version_id);
  bool has_column = false;
  if (isPersistent(table_ptr_)) {
    has_column = table_ptr_->hasColumn(input_attribute_name);
  } else {
    has_column = table_ptr_->hasColumn(fully_qualified_col);
  }
  if (!has_column) {
    table_ptr_->printSchema();
    COGADB_FATAL_ERROR("Column " << fully_qualified_col << " not found "
                                 << " in table " << table_ptr_->getName()
                                 << "!",
                       "");
  }
  /* automatically derive the type of the attribute */
  std::pair<bool, AttributeType> ret;
  if (isPersistent(table_ptr_)) {
    ret = ::CoGaDB::getAttributeType(table_ptr_, input_attribute_name);
  } else {
    ret = ::CoGaDB::getAttributeType(table_ptr_, fully_qualified_col);
  }
  if (!ret.first) {
    COGADB_FATAL_ERROR("Could not retrieve type for attribute reference: "
                           << table_ptr_->getName() << "."
                           << input_attribute_name << "." << version_id,
                       "");
  }
  assert(ret.first == true);
  type_ = ret.second;
  /* If the user did not specifiy a result name, then the parameter was
   * default constructed. If it is empty, take input attribute name as
   * result attribute name. */
  if (_result_attribute_name.empty()) {
    //            result_attribute_name = table_ptr_->getName();
    //            result_attribute_name+=".";
    result_attribute_name += input_attribute_name;
    result_attribute_name += ".";
    result_attribute_name += boost::lexical_cast<std::string>(version_id);
  }
}

AttributeReference::AttributeReference(
    const std::string& _input_attribute_name, const AttributeType& _type,
    const std::string& _result_attribute_name, uint32_t _version_id)
    : table_ptr_(),
      input_attribute_name(_input_attribute_name),
      type_(_type),
      version_id(_version_id),
      result_attribute_name(_result_attribute_name),
      attr_ref_type(COMPUTED_ATTRIBUTE) {}

AttributeReference::AttributeReference(const AttributeReference& other)
    : table_ptr_(other.table_ptr_),
      input_attribute_name(other.input_attribute_name),
      type_(other.type_),
      version_id(other.version_id),
      result_attribute_name(other.result_attribute_name),
      attr_ref_type(other.attr_ref_type) {}

AttributeReference& AttributeReference::operator=(
    const AttributeReference& other) {
  if (this != &other)  // protect against invalid self-assignment
  {
    table_ptr_ = other.table_ptr_;
    input_attribute_name = other.input_attribute_name;
    type_ = other.type_;
    result_attribute_name = other.result_attribute_name;
    version_id = other.version_id;
    attr_ref_type = other.attr_ref_type;
  }
  return *this;
}

const std::string AttributeReference::getUnversionedTableName() const {
  if (attr_ref_type == INPUT_ATTRIBUTE) {
    return table_ptr_->getName();
  } else {
    return std::string("");
  }
}

const std::string AttributeReference::getVersionedTableName() const {
  std::stringstream ss;
  ss << getUnversionedTableName() << version_id;
  return ss.str();
}

const std::string AttributeReference::getUnversionedAttributeName() const {
  return input_attribute_name;
}

const std::string AttributeReference::getVersionedAttributeName() const {
  //        if(attr_ref_type==INPUT_ATTRIBUTE){
  std::stringstream ss;
  ss << input_attribute_name;
  if (attr_ref_type == INPUT_ATTRIBUTE) ss << version_id;
  return ss.str();
  //        }else{
  //            return std::string("");
  //        }
}

const std::string AttributeReference::getResultAttributeName() const noexcept {
  return result_attribute_name;
}

AttributeType AttributeReference::getAttributeType() const noexcept {
  return type_;
}

AttributeReferenceType AttributeReference::getAttributeReferenceType() const
    noexcept {
  return attr_ref_type;
}

const TablePtr AttributeReference::getTable() const { return table_ptr_; }

const ColumnPtr AttributeReference::getColumn() const {
  if (isPersistent(table_ptr_)) {
    return table_ptr_->getColumnbyName(getUnversionedAttributeName());
  } else {
    return table_ptr_->getColumnbyName(CoGaDB::toString(*this));
  }
}

const HashTablePtr AttributeReference::getHashTable() const {
  return table_ptr_->getHashTablebyName(
      createFullyQualifiedColumnIdentifier(*this));
}

bool AttributeReference::hasHashTable() const {
  return (table_ptr_->getHashTablebyName(
              createFullyQualifiedColumnIdentifier(*this)) != NULL);
}

uint32_t AttributeReference::getVersion() const { return version_id; }

void AttributeReference::setVersion(uint32_t version) {
  this->version_id = version;
}

void AttributeReference::setTable(TablePtr table) { table_ptr_ = table; }

bool AttributeReference::operator==(const AttributeReference& other) const {
  return other.result_attribute_name == result_attribute_name &&
         other.input_attribute_name == input_attribute_name &&
         (other.table_ptr_ == table_ptr_ ||
          other.table_ptr_->getName() == table_ptr_->getName());
}

const AttributeReferencePtr createInputAttributeForNewTable(
    const AttributeReference& attr, const TablePtr table) {
  if (!table) return AttributeReferencePtr();
  return boost::make_shared<AttributeReference>(
      table, attr.getUnversionedAttributeName(), attr.getResultAttributeName(),
      attr.getVersion());
}

const AttributeReferencePtr createInputAttribute(
    const TablePtr table, const std::string& input_attribute_name,
    const std::string& result_attribute_name, const uint32_t version_id) {
  if (result_attribute_name.empty()) {
    return boost::make_shared<AttributeReference>(table, input_attribute_name,
                                                  std::string(), version_id);
  } else {
    return boost::make_shared<AttributeReference>(
        table, input_attribute_name, result_attribute_name, version_id);
  }
}

const AttributeReferencePtr createComputedAttribute() {
  return AttributeReferencePtr();
}

bool getColumnProperties(const TablePtr table, const std::string& column_name,
                         ColumnProperties& ret_col_props) {
  if (table) {
    std::vector<ColumnProperties> col_props = table->getPropertiesOfColumns();
    for (size_t i = 0; i < col_props.size(); ++i) {
      if (compareAttributeReferenceNames(column_name, col_props[i].name)) {
        ret_col_props = col_props[i];
        return true;
      }
    }
  }
  return false;
}

bool getColumnProperties(const AttributeReference& attr,
                         ColumnProperties& ret_col_props) {
  if (!isComputed(attr)) {
    if (isPersistent(attr.getTable())) {
      return getColumnProperties(
          attr.getTable(), attr.getUnversionedAttributeName(), ret_col_props);
    } else {
      return getColumnProperties(attr.getTable(), CoGaDB::toString(attr),
                                 ret_col_props);
    }
  } else {
    return false;
  }
}

const AttributeReferencePtr getAttributeReference(
    const std::string& column_name, uint32_t version) {
  if (DataDictionary::instance().existColumn(column_name)) {
    std::list<std::pair<ColumnPtr, TablePtr> > cols =
        DataDictionary::instance().getColumnsforColumnName(column_name);
    if (cols.size() > 1) {
      COGADB_FATAL_ERROR(
          "Ambiguous column name detected: '" << column_name << "'", "");
    } else {
      TablePtr table = cols.front().second;
      return createInputAttribute(table, column_name, column_name, version);
    }
  } else {
    return AttributeReferencePtr();
  }
}

bool isComputed(const AttributeReference& attr) {
  return (attr.getAttributeReferenceType() == COMPUTED_ATTRIBUTE);
}
bool isInputAttribute(const AttributeReference& attr) {
  return (attr.getAttributeReferenceType() == INPUT_ATTRIBUTE);
}

AttributeType getAttributeType(const AttributeReference& attr) {
  return attr.getAttributeType();
}

ColumnType getColumnType(const AttributeReference& attr) {
  if (isComputed(attr)) {
    if (getAttributeType(attr) == VARCHAR) {
      return DICTIONARY_COMPRESSED;
    } else {
      return PLAIN_MATERIALIZED;
    }
  } else {
    return attr.getColumn()->getColumnType();
  }
}

bool isPrimaryKey(const AttributeReference& attr) {
  if (attr.getTable()) {
    return attr.getTable()->hasPrimaryKeyConstraint(
        attr.getUnversionedAttributeName());
  }
  return false;
}
bool isForeignKey(const AttributeReference& attr) {
  if (attr.getTable()) {
    return attr.getTable()->hasForeignKeyConstraint(
        attr.getUnversionedAttributeName());
  }
  return false;
}
const AttributeReferencePtr getForeignKeyAttribute(
    const AttributeReference& attr) {
  if (isForeignKey(attr)) {
    assert(attr.getTable() != NULL);
    const ForeignKeyConstraint* foreign_key_constr =
        attr.getTable()->getForeignKeyConstraint(
            attr.getUnversionedAttributeName());
    TablePtr foreign_key_table =
        getTablebyName(foreign_key_constr->getNameOfForeignKeyTable());
    assert(foreign_key_table != NULL);

    return boost::make_shared<AttributeReference>(
        foreign_key_table, foreign_key_constr->getNameOfForeignKeyColumn(),
        foreign_key_constr->getNameOfForeignKeyColumn(), attr.getVersion());
  } else {
    return AttributeReferencePtr();
  }
}

bool areStatisticsUpToDate(const AttributeReference& attr) {
  ColumnProperties col_props;
  if (!getColumnProperties(attr, col_props)) {
    return false;
  }
  return col_props.statistics_up_to_date;
}

size_t getNumberOfRequiredBits(const AttributeReference& attr) {
  if (!isComputed(attr)) {
    if (getColumnType(attr) == DICTIONARY_COMPRESSED) {
      ColumnPtr col = attr.getColumn();
      assert(col != NULL);
      DictionaryCompressedCol* dict_compressed =
          dynamic_cast<DictionaryCompressedCol*>(col.get());
      assert(dict_compressed != NULL);
      uint64_t largest_id = dict_compressed->getLargestID();
      return getGreaterPowerOfTwo(largest_id);
    }

    return attr.getColumn()->getNumberOfRequiredBits();
  } else {
    /* check common types of computed attributes */
    if (attr.getAttributeType() == DOUBLE) {
      return sizeof(double) * 8;
    } else if (attr.getAttributeType() == FLOAT) {
      return sizeof(float) * 8;
    } else if (attr.getAttributeType() == INT) {
      return sizeof(int32_t) * 8;
    } else if (attr.getAttributeType() == UINT32) {
      return sizeof(uint32_t) * 8;
    } else if (attr.getAttributeType() == OID) {
      return sizeof(TID) * 8;
    } else {
      /* we do not know what this could be, wset to 65 to flag that we
       cannot apply bitbpacking for group by for this attribute */
      return 65;
    }
  }
}

bool isSortedAscending(const AttributeReference& attr) {
  ColumnProperties col_props;
  if (!getColumnProperties(attr, col_props)) {
    return false;
  }
  return col_props.is_sorted_ascending;
}

bool isSortedDescending(const AttributeReference& attr) {
  ColumnProperties col_props;
  if (!getColumnProperties(attr, col_props)) {
    return false;
  }
  return col_props.is_sorted_descending;
}

bool isDenseValueArrayStartingWithZero(const AttributeReference& attr) {
  ColumnProperties col_props;
  if (!getColumnProperties(attr, col_props)) {
    return false;
  }
  return col_props.is_dense_value_array_starting_with_zero;
}

const std::string toString(const AttributeReference& attr) {
  std::stringstream ss;
  if (isInputAttribute(attr)) {
    ss << attr.getUnversionedAttributeName() << "." << attr.getVersion();
  } else {
    ss << "TEMPORARY." << attr.getVersionedAttributeName();
  }
  return ss.str();
}

void to_upper(const std::string& lower, std::size_t start, std::size_t end,
              std::string& result) {
  result.resize(end - start);

  std::string::const_iterator begin_itr(lower.begin()), end_itr(lower.begin());
  std::advance(begin_itr, start);
  std::advance(end_itr, end);

  // http://stackoverflow.com/questions/7131858/stdtransform-and-toupper-no-matching-function
  std::transform(begin_itr, end_itr, result.begin(),
                 static_cast<int (*)(int)>(std::toupper));
}

unsigned int find_all_characters(const std::string& search, char find,
                                 unsigned int max_pos_count = 0,
                                 std::size_t positions[] = nullptr) {
  unsigned int found = 0;

  for (std::size_t last_pos = search.find(find); last_pos != std::string::npos;
       last_pos = search.find(find, last_pos + 1), ++found) {
    if (found < max_pos_count) {
      positions[found] = last_pos;
    }
  }

  return found;
}

bool check_for_brackets(const std::string& column_identifier) {
  assert(find_all_characters(column_identifier, '(') ==
         find_all_characters(column_identifier, ')'));
  return column_identifier.find('(') != std::string::npos ||
         column_identifier.find(')') != std::string::npos;
}

bool parseColumnIndentifierName(const std::string& column_identifier,
                                std::string& table_name,
                                std::string& attribute_name,
                                uint32_t& version) {
  const unsigned int max_point_count = 2;
  std::size_t point_positions[max_point_count];
  unsigned int point_count = find_all_characters(
      column_identifier, '.', max_point_count, point_positions);

  if (check_for_brackets(column_identifier)) {
    return false;
  } else if (point_count == 0) {
    to_upper(column_identifier, 0, column_identifier.size(), attribute_name);
    version = 1;

    size_t num_occurences =
        DataDictionary::instance().countNumberOfOccurencesOfColumnName(
            attribute_name);
    if (num_occurences == 0) {
      return false;
    } else if (num_occurences > 1) {
      COGADB_FATAL_ERROR("Column name '" << column_identifier << "' ambiguous!",
                         "");
    } else {
      TablePtr table =
          DataDictionary::instance().getTableForColumnName(attribute_name);
      assert(table != NULL);
      table_name = table->getName();
      return true;
    }
  } else if (point_count == 1) {
    to_upper(column_identifier, 0, point_positions[0], table_name);
    to_upper(column_identifier, point_positions[0] + 1,
             column_identifier.size(), attribute_name);
    version = 1;
  } else if (point_count == 2) {
    to_upper(column_identifier, 0, point_positions[0], table_name);
    to_upper(column_identifier, point_positions[0] + 1, point_positions[1],
             attribute_name);

    std::size_t version_start = point_positions[1] + 1;
    std::size_t version_count = column_identifier.size() - version_start;
    try {
      version = boost::lexical_cast<uint32_t>(
          column_identifier.c_str() + version_start, version_count);
    } catch (boost::bad_lexical_cast&) {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

bool isFullyQualifiedColumnIdentifier(const std::string& column_identifier) {
  const unsigned int max_point_count = 2;
  std::size_t point_positions[max_point_count];
  unsigned int point_count = find_all_characters(
      column_identifier, '.', max_point_count, point_positions);

  if (check_for_brackets(column_identifier)) {
    return false;
  } else if (point_count == 2) {
    std::size_t digit_pos(point_positions[1] + 1),
        end(column_identifier.size());

    bool result = true;
    for (; digit_pos < end; ++digit_pos) {
      result &= std::isdigit(column_identifier[digit_pos]);
    }

    return result;
  } else {
    return false;
  }
}

bool convertColumnNameToFullyQualifiedName(const std::string& column_name,
                                           std::string& fully_qualified_name) {
  std::string attribute_type_identifier;
  std::string attribute_name;
  uint32_t version = 1;
  if (!parseColumnIndentifierName(column_name, attribute_type_identifier,
                                  attribute_name, version)) {
    return false;
  }
  fully_qualified_name = attribute_type_identifier;
  fully_qualified_name += ".";
  fully_qualified_name += attribute_name;
  fully_qualified_name += ".";
  fully_qualified_name += boost::lexical_cast<std::string>(version);
  return true;
}

const std::string convertToFullyQualifiedNameIfRequired(
    const std::string& column_name) {
  std::string qualified_column_name = column_name;
  if (!isFullyQualifiedColumnIdentifier(column_name)) {
    if (!convertColumnNameToFullyQualifiedName(column_name,
                                               qualified_column_name)) {
    }
  }
  return qualified_column_name;
}

const std::string replaceTableNameInQualifiedColumnIdentifier(
    const std::string& column_identifier, const std::string& new_table_name) {
  std::string table_name;
  std::string attribute_name;
  uint32_t version = 1;
  if (!parseColumnIndentifierName(column_identifier, table_name, attribute_name,
                                  version)) {
    return column_identifier;
  }
  std::cout << "Transform: '" << column_identifier << "'=>'"
            << createFullyQualifiedColumnIdentifier(new_table_name,
                                                    attribute_name, version)
            << "'" << std::endl;
  return createFullyQualifiedColumnIdentifier(new_table_name, attribute_name,
                                              version);
}

const AttributeReferencePtr getAttributeFromColumnIdentifier(
    const std::string& column_identifier) {
  std::string attribute_type_identifier;
  std::string attribute_name;
  uint32_t version = 1;
  if (!parseColumnIndentifierName(column_identifier, attribute_type_identifier,
                                  attribute_name, version)) {
    return AttributeReferencePtr();
  }
  TablePtr table = getTablebyName(attribute_type_identifier);
  if (!table) {
    return AttributeReferencePtr();
  }
  AttributeReferencePtr attr(new AttributeReference(
      table, attribute_type_identifier + std::string(".") + attribute_name,
      createFullyQualifiedColumnIdentifier(table->getName(), attribute_name,
                                           version),
      version));
  return attr;
}

const std::string createFullyQualifiedColumnIdentifier(
    const std::string& table_name, const std::string& attribute_name,
    const uint32_t& version) {
  std::stringstream name;
  name << table_name << "." << attribute_name << "." << version;
  return name.str();
}

const std::string createFullyQualifiedColumnIdentifier(
    const AttributeReferencePtr attr) {
  assert(attr != NULL);
  return createFullyQualifiedColumnIdentifier(*attr);
}

bool isPlainAttributeName(const std::string& attribute_name) {
  if (boost::count(attribute_name, '.') == 0) {
    return true;
  } else {
    return false;
  }
}

const std::string createFullyQualifiedColumnIdentifier(
    const AttributeReference& attr) {
  if (isPlainAttributeName(attr.getUnversionedAttributeName())) {
    return createFullyQualifiedColumnIdentifier(
        attr.getUnversionedTableName(), attr.getUnversionedAttributeName(),
        attr.getVersion());
  } else {
    std::string original_table_name;
    std::string attribute_name;
    uint32_t version;
    if (!parseColumnIndentifierName(attr.getUnversionedAttributeName(),
                                    original_table_name, attribute_name,
                                    version)) {
      COGADB_FATAL_ERROR(
          "Could not parse attribute identifier from attribute reference!", "");
    }
    version = attr.getVersion();
    /* we need to use the original table name as table qualifier! */
    return createFullyQualifiedColumnIdentifier(
        original_table_name,
        attribute_name,  // qualified_name,
        attr.getVersion());
  }
}

bool compareAttributeReferenceNames(const std::string& lhs,
                                    const std::string& rhs) {
  std::string fully_qualified_name_lhs = lhs;
  std::string fully_qualified_name_rhs = rhs;
  if (!isFullyQualifiedColumnIdentifier(lhs)) {
    if (!convertColumnNameToFullyQualifiedName(lhs, fully_qualified_name_lhs)) {
    }
  }

  if (!isFullyQualifiedColumnIdentifier(rhs)) {
    if (!convertColumnNameToFullyQualifiedName(rhs, fully_qualified_name_rhs)) {
    }
  }

  return (fully_qualified_name_lhs == fully_qualified_name_rhs);
}

const std::string getAttributeNameFromColumnIdentifier(
    const std::string& column_identifier) {
  std::vector<std::string> tokens;
  boost::split(tokens, column_identifier, boost::is_any_of("."));

  if (tokens.size() == 1) {
    return column_identifier;
  } else if (tokens.size() == 2 || tokens.size() == 3) {
    return tokens[1];
  } else {
    COGADB_FATAL_ERROR("", "");
    return column_identifier;
  }
}

bool getVersionFromColumnIdentifier(const std::string& column_identifier,
                                    uint32_t& version) {
  std::vector<std::string> tokens;
  boost::split(tokens, column_identifier, boost::is_any_of("."));

  if (tokens.size() == 3) {
    try {
      version = boost::lexical_cast<uint32_t>(tokens[2]);
      return true;
    } catch (const boost::bad_lexical_cast&) {
      return false;
    }
  }
  return false;
}

}  // end namespace CoGaDB
