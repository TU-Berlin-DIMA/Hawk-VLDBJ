#include <compression/bit_vector_compressed_column.hpp>
#include <compression/bitpacked_dictionary_compressed_column.hpp>
#include <compression/delta_compressed_column.hpp>
#include <compression/dictionary_compressed_column.hpp>
#include <compression/rle_compressed_column.hpp>
#include <compression/rle_delta_one_compressed_column_int.hpp>
#include <core/base_column.hpp>
#include <core/column.hpp>
#include <core/column_base_typed.hpp>
#include <core/compressed_column.hpp>
#include <core/global_definitions.hpp>
#include <string>

#include <boost/filesystem.hpp>

#include <core/base_table.hpp>
#include <core/runtime_configuration.hpp>
#include <core/table.hpp>

#include <boost/generator_iterator.hpp>
#include <boost/random.hpp>

namespace CoGaDB {
namespace unit_tests {

using namespace std;

bool unittest(boost::shared_ptr<ColumnBaseTyped<int> > ptr);
bool unittest(boost::shared_ptr<ColumnBaseTyped<float> > ptr);
bool unittest(boost::shared_ptr<ColumnBaseTyped<std::string> > ptr);

bool delta();
bool bit_vector();
bool run_length();
bool dictionary_compression();
bool bitpacked_dictionary_compression();
bool run_length_delta_one_number();
bool bulk_test_update(boost::shared_ptr<CompressedColumn<int> > compressed_col,
                      boost::shared_ptr<Column<int> > uncompressed_col);
bool bulk_test_delete(boost::shared_ptr<CompressedColumn<int> > compressed_col,
                      boost::shared_ptr<Column<int> > uncompressed_col);
bool unittests(boost::shared_ptr<CompressedColumn<int> > col,
               boost::shared_ptr<CompressedColumn<float> > col_float,
               boost::shared_ptr<CompressedColumn<std::string> > col_string);

bool compressioned_columns_tests() {
  long cur_time = time(NULL);
  // cur_time = 1372344200;
  std::cout << cur_time;
  srand(cur_time);
  if (!run_length()) {
    return false;
  }
  if (!delta()) {
    return false;
  }
  if (!dictionary_compression()) {
    return false;
  }
  if (!bitpacked_dictionary_compression()) {
    return false;
  }
  if (!run_length_delta_one_number()) {
    return false;
  }

  if (!bit_vector()) {
    return false;
  }
  return true;
}

bool delta() {
  boost::shared_ptr<CompressedColumn<int> > col(
      new DeltaCompressedColumn<int>("int column delta", INT));
  boost::shared_ptr<CompressedColumn<float> > col_float(
      new DeltaCompressedColumn<float>("float column delta", FLOAT));
  boost::shared_ptr<CompressedColumn<std::string> > col_string(
      new DeltaCompressedColumn<std::string>("string column delta", VARCHAR));

  boost::shared_ptr<DeltaCompressedColumn<int> > compressed_col(
      new DeltaCompressedColumn<int>("int column delta", INT));

  boost::shared_ptr<Column<int> > uncompressed_col(
      new Column<int>("int column", INT));
  std::vector<int> reference_data(100);

  std::cout << std::endl
            << "TESTING DELTA COMPRESSED COLUMN..." << std::endl
            << std::endl;

  if (!unittests(col, col_float, col_string)) {
    return false;
  }

  std::cout << "BULK UPDATE TEST...";  // << std::endl;
  unsigned int i = 0;
  for (; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_update(compressed_col, uncompressed_col)) {
    return false;
  }

  std::cout << "BULK DELETE TEST...";  // << std::endl;

  compressed_col->clearContent();
  uncompressed_col->clearContent();

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_delete(compressed_col, uncompressed_col)) {
    return false;
  }
  return true;
}

bool bit_vector() {
  boost::shared_ptr<CompressedColumn<int> > col(
      new BitVectorCompressedColumn<int>("int column bit vector", INT));
  boost::shared_ptr<CompressedColumn<float> > col_float(
      new BitVectorCompressedColumn<float>("float column bit vector", FLOAT));
  boost::shared_ptr<CompressedColumn<std::string> > col_string(
      new BitVectorCompressedColumn<std::string>("string column bit vector",
                                                 VARCHAR));

  boost::shared_ptr<BitVectorCompressedColumn<int> > compressed_col(
      new BitVectorCompressedColumn<int>("int column bit vector", INT));

  boost::shared_ptr<Column<int> > uncompressed_col(
      new Column<int>("int column", INT));
  std::vector<int> reference_data(100);

  std::cout << std::endl
            << "TESTING BITVECTOR COMPRESSED COLUMN..." << std::endl
            << std::endl;

  if (!unittests(col, col_float, col_string)) {
    return false;
  }

  std::cout << "BULK UPDATE TEST...";  // << std::endl;

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_update(compressed_col, uncompressed_col)) {
    return false;
  }

  std::cout << "BULK DELETE TEST...";  // << std::endl;

  compressed_col->clearContent();
  uncompressed_col->clearContent();

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_delete(compressed_col, uncompressed_col)) {
    return false;
  }
  return true;
}

bool run_length() {
  boost::shared_ptr<CompressedColumn<int> > col(
      new RLECompressedColumn<int>("int column run length", INT));
  boost::shared_ptr<CompressedColumn<float> > col_float(
      new RLECompressedColumn<float>("float column run length", FLOAT));
  boost::shared_ptr<CompressedColumn<std::string> > col_string(
      new RLECompressedColumn<std::string>("string column run length",
                                           VARCHAR));

  boost::shared_ptr<RLECompressedColumn<int> > compressed_col(
      new RLECompressedColumn<int>("int column run length", INT));

  boost::shared_ptr<Column<int> > uncompressed_col(
      new Column<int>("int column", INT));
  std::vector<int> reference_data(100);

  std::cout << std::endl
            << "TESTING RUN LENGTH COMPRESSED COLUMN..." << std::endl
            << std::endl;

  if (!unittests(col, col_float, col_string)) {
    return false;
  }

  std::cout << "BULK UPDATE TEST...";  // << std::endl;

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_update(compressed_col, uncompressed_col)) {
    return false;
  }

  std::cout << "BULK DELETE TEST...";  // << std::endl;

  compressed_col->clearContent();
  uncompressed_col->clearContent();

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_delete(compressed_col, uncompressed_col)) {
    return false;
  }
  return true;
}

bool dictionary_compression() {
  boost::shared_ptr<CompressedColumn<int> > col(
      new DictionaryCompressedColumn<int>("int column dict", INT));
  boost::shared_ptr<CompressedColumn<float> > col_float(
      new DictionaryCompressedColumn<float>("float column dict", FLOAT));
  boost::shared_ptr<CompressedColumn<std::string> > col_string(
      new DictionaryCompressedColumn<std::string>("string column dict",
                                                  VARCHAR));

  boost::shared_ptr<DictionaryCompressedColumn<int> > compressed_col(
      new DictionaryCompressedColumn<int>("int column dictionary encoded",
                                          INT));

  boost::shared_ptr<Column<int> > uncompressed_col(
      new Column<int>("int column", INT));
  std::vector<int> reference_data(100);

  std::cout << std::endl
            << "TESTING DICTIONARY COMPRESSED COLUMN..." << std::endl
            << std::endl;

  if (!unittests(col, col_float, col_string)) {
    return false;
  }

  std::cout << "BULK UPDATE TEST...";  // << std::endl;

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_update(compressed_col, uncompressed_col)) {
    return false;
  }

  std::cout << "BULK DELETE TEST...";  // << std::endl;

  compressed_col->clearContent();
  uncompressed_col->clearContent();

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_delete(compressed_col, uncompressed_col)) {
    return false;
  }
  return true;
}

bool bitpacked_dictionary_compression() {
  boost::shared_ptr<CompressedColumn<int> > col(
      new BitPackedDictionaryCompressedColumn<int>("int column bit packed dict",
                                                   INT, 7));
  boost::shared_ptr<CompressedColumn<float> > col_float(
      new BitPackedDictionaryCompressedColumn<float>(
          "float column bit packed dict", FLOAT, 7));
  boost::shared_ptr<CompressedColumn<std::string> > col_string(
      new BitPackedDictionaryCompressedColumn<std::string>(
          "string column bit packed dict", VARCHAR, 7));

  boost::shared_ptr<BitPackedDictionaryCompressedColumn<int> > compressed_col(
      new BitPackedDictionaryCompressedColumn<int>(
          "int column bit packed dictionary encoded", INT, 7));

  boost::shared_ptr<Column<int> > uncompressed_col(
      new Column<int>("int column", INT));
  std::vector<int> reference_data(100);

  std::cout << std::endl
            << "TESTING BIT PACKED DICTIONARY COMPRESSED COLUMN..." << std::endl
            << std::endl;

  if (!unittests(col, col_float, col_string)) {
    return false;
  }

  /* TODO
  std::cout << "BULK UPDATE TEST..."; // << std::endl;

  for (unsigned int i = 0; i < reference_data.size(); i++) {
      reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_update(compressed_col, uncompressed_col)) {
      return false;
  }

  std::cout << "BULK DELETE TEST..."; // << std::endl;

  compressed_col->clearContent();
  uncompressed_col->clearContent();

  for (unsigned int i = 0; i < reference_data.size(); i++) {
      reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());


  if (!bulk_test_delete(compressed_col, uncompressed_col)) {
      return false;
  }
   */
  return true;
}

bool run_length_delta_one_number() {
  boost::shared_ptr<CompressedColumn<int> > col(
      new RLEDeltaOneCompressedColumnNumber<int>("int column run length", INT));
  // TODO better modularization of unit tests, for now we use the normal RLE
  // column to not break the tests
  boost::shared_ptr<CompressedColumn<float> > col_float(
      new RLECompressedColumn<float>("float column run length", FLOAT));
  boost::shared_ptr<CompressedColumn<std::string> > col_string(
      new RLECompressedColumn<std::string>("string column run length",
                                           VARCHAR));

  boost::shared_ptr<RLECompressedColumn<int> > compressed_col(
      new RLECompressedColumn<int>("int column run length", INT));

  boost::shared_ptr<Column<int> > uncompressed_col(
      new Column<int>("int column", INT));
  std::vector<int> reference_data(100);

  std::cout << std::endl
            << "TESTING RUN LENGTH DELTA ONE COMPRESSED NUMBER COLUMN..."
            << std::endl
            << std::endl;

  if (!unittests(col, col_float, col_string)) {
    return false;
  }

  std::cout << "BULK UPDATE TEST...";  // << std::endl;

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_update(compressed_col, uncompressed_col)) {
    return false;
  }

  std::cout << "BULK DELETE TEST...";  // << std::endl;

  compressed_col->clearContent();
  uncompressed_col->clearContent();

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = rand() % 100;
  }

  uncompressed_col->insert(reference_data.begin(), reference_data.end());
  compressed_col->insert(reference_data.begin(), reference_data.end());

  if (!bulk_test_delete(compressed_col, uncompressed_col)) {
    return false;
  }
  return true;
}

bool unittests(boost::shared_ptr<CompressedColumn<int> > col,
               boost::shared_ptr<CompressedColumn<float> > col_float,
               boost::shared_ptr<CompressedColumn<std::string> > col_string) {
  if (!unittest(col)) {
    std::cout << "At least one Unittest Failed!" << std::endl;
    return false;
  }
  std::cout << "Unitests Passed!" << std::endl << std::endl;

  if (!unittest(col_float)) {
    std::cout << "At least one Unittest Failed!" << std::endl;
    return false;
  }
  std::cout << "Unitests Passed!" << std::endl << std::endl;

  if (!unittest(col_string)) {
    std::cout << "At least one Unittest Failed!" << std::endl;
    return false;
  }
  std::cout << "Unitests Passed!" << std::endl << std::endl;

  return true;
}

/****** BULK UPDATE TEST ******/
bool bulk_test_update(boost::shared_ptr<CompressedColumn<int> > compressed_col,
                      boost::shared_ptr<Column<int> > uncompressed_col) {
  bool result =
      *(boost::static_pointer_cast<ColumnBaseTyped<int> >(uncompressed_col)) ==
      *(boost::static_pointer_cast<ColumnBaseTyped<int> >(compressed_col));
  if (!result) {
    std::cerr << std::endl << "operator== TEST FAILED!" << std::endl;
    return false;
  }
  PositionListPtr tids = createPositionList();
  int new_value = rand() % 100;
  for (unsigned int i = 0; i < 10; i++) {
    tids->push_back(rand() % uncompressed_col->size());
  }

  uncompressed_col->update(tids, new_value);
  compressed_col->update(tids, new_value);

  result =
      *(boost::static_pointer_cast<ColumnBaseTyped<int> >(uncompressed_col)) ==
      *(boost::static_pointer_cast<ColumnBaseTyped<int> >(compressed_col));
  if (!result) {
    std::cerr << std::endl << "BULK UPDATE TEST FAILED!" << std::endl;
    std::cout << "uncompressed column: " << std::endl;
    uncompressed_col->print();
    std::cout << "compressed column: " << std::endl;
    compressed_col->print();
    return false;
  }
  std::cout << "SUCCESS" << std::endl;
  return true;
}

/****** BULK DELETE TEST ******/
bool bulk_test_delete(boost::shared_ptr<CompressedColumn<int> > compressed_col,
                      boost::shared_ptr<Column<int> > uncompressed_col) {
  bool result =
      *(boost::static_pointer_cast<ColumnBaseTyped<int> >(uncompressed_col)) ==
      *(boost::static_pointer_cast<ColumnBaseTyped<int> >(compressed_col));
  if (!result) {
    std::cerr << std::endl << "operator== TEST FAILED!" << std::endl;
    return false;
  }

  PositionListPtr tids = createPositionList();

  for (unsigned int i = 0; i < 10; i++) {
    tids->push_back(rand() % uncompressed_col->size());
  }
  std::sort(tids->begin(), tids->end());
  //	TID last_tid = tids->back();
  //	for(int i = 8; i >= 0; --i){
  //		TID cur_tid = (*tids)[i];
  //		if (cur_tid == last_tid) {
  //			tids->erase(tids->begin() + i);
  //		}
  //		last_tid = cur_tid;
  //	}

  compressed_col->remove(tids);
  uncompressed_col->remove(tids);

  result =
      *(boost::static_pointer_cast<ColumnBaseTyped<int> >(uncompressed_col)) ==
      *(boost::static_pointer_cast<ColumnBaseTyped<int> >(compressed_col));
  if (!result) {
    std::cerr << "BULK DELETE TEST FAILED!" << std::endl;
    return false;
  }
  std::cout << "SUCCESS" << std::endl;
  return true;
}

template <typename T>
const T get_rand_value() {
  return 0;
}

template <>
const int get_rand_value() {
  return rand() % 100;
}

template <>
const float get_rand_value() {
  return float(rand() % 10000) / 100;
}

template <>
const std::string get_rand_value() {
  std::string characterfield = "abcdefghijklmnopqrstuvwxyz";

  std::string s;
  for (unsigned int i = 0; i < 10; i++) {
    s.push_back(characterfield[rand() % characterfield.size()]);
  }
  return s;
}

template <class T>
void fill_column(boost::shared_ptr<ColumnBaseTyped<T> > col,
                 std::vector<T>& reference_data) {
  for (unsigned int i = 0; i < reference_data.size(); i++) {
    reference_data[i] = get_rand_value<T>();
  }

  for (unsigned int i = 0; i < reference_data.size(); i++) {
    col->insert(reference_data[i]);
  }

  std::cout << "Size in Bytes: " << col->getSizeinBytes() << std::endl;
}

template <class T>
bool equals(std::vector<T> reference_data,
            boost::shared_ptr<ColumnBaseTyped<T> > col) {
  for (unsigned int i = 0; i < reference_data.size() - 1; i++) {
    T col_value = (*col)[i];
    T ref_value = reference_data[i];
    if (ref_value != col_value) {
      std::cout << "Fatal Error! In Unittest: read invalid data" << std::endl;
      std::cout << "Column: '" << col->getName() << "' TID: '" << i
                << "' Expected Value: '" << reference_data[i]
                << "' Actual Value: '" << col_value << "'" << std::endl;
      return false;
    }
  }
  return true;
}

template <class T>
bool test_column(boost::shared_ptr<ColumnBaseTyped<T> > col,
                 std::vector<T>& reference_data) {
  /****** BASIC INSERT TEST ******/
  std::cout
      << "BASIC INSERT TEST: Filling column with data...";  // << std::endl;
  // col->insert(reference_data.begin(),reference_data.end());

  if (reference_data.size() != col->size()) {
    std::cout << "Fatal Error! In Unittest: invalid data size" << std::endl;
    return false;
  }

  if (!equals(reference_data, col)) {
    std::cerr << "BASIC INSERT TEST FAILED!" << std::endl;
    return false;
  }

  std::cout << std::endl;

  std::cout << "SUCCESS" << std::endl;
  /****** VIRTUAL COPY CONSTRUCTOR TEST ******/
  std::cout << "VIRTUAL COPY CONSTRUCTOR TEST...";

  // boost::shared_ptr<DictionaryCompressedColumn<int> > compressed_col (new
  // DictionaryCompressedColumn<int>("compressed int column",INT));
  // compressed_col->insert(reference_data.begin(),reference_data.end());

  ColumnPtr copy = col->copy();
  if (!copy) {
    std::cerr << std::endl
              << "VIRTUAL COPY CONSTRUCTOR TEST FAILED!" << std::endl;
    return false;
  }
  bool result = *(boost::static_pointer_cast<ColumnBaseTyped<T> >(copy)) ==
                *(boost::static_pointer_cast<ColumnBaseTyped<T> >(col));
  if (!result) {
    std::cerr << std::endl
              << "VIRTUAL COPY CONSTRUCTOR TEST FAILED!" << std::endl;
    return false;
  }
  std::cout << "SUCCESS" << std::endl;
  /****** UPDATE TEST ******/
  TID tid = rand() % 100;
  T new_value = get_rand_value<T>();
  std::cout << "UPDATE TEST: Update value on Position '" << tid
            << "' to new value '" << new_value << "'...";  // << std::endl;

  reference_data[tid] = new_value;

  col->update(tid, new_value);

  if (!equals(reference_data, col)) {
    std::cerr << "UPDATE TEST FAILED!" << std::endl;
    return false;
  }
  std::cout << "SUCCESS" << std::endl;
  /****** DELETE TEST ******/
  {
    TID tid = rand() % 100;

    std::cout << "DELETE TEST: Delete value on Position '" << tid
              << "'...";  // << std::endl;

    /*
    // ############# Just for debugging!! ############
    for (unsigned int i = 0; i < reference_data.size(); i++) {
        std::cout << "Element " << i << ": " << reference_data[i] << "(" << i
                << ") ";
    }
    std::cout << std::endl;
    col->print();
    // ############# Just for debugging!! ############
     */

    reference_data.erase(reference_data.begin() + tid);

    col->remove(tid);

    if (!equals(reference_data, col)) {
      std::cerr << "DELETE TEST FAILED!" << std::endl;
      return false;
    }
    std::cout << "SUCCESS" << std::endl;
  }
  /****** STORE AND LOAD TEST ******/
  {
    std::cout << "STORE AND LOAD TEST: store column data on disc and load "
                 "it...";  // << std::endl;

    boost::filesystem::create_directories("unittest_data");
    col->store("unittest_data/");

    col->clearContent();
    if (col->size() != 0) {
      std::cout << "Fatal Error! 'col->size()' returned non zero after call to "
                   "'col->clearContent()'\nTEST FAILED"
                << std::endl;
      return false;
    }

    // boost::shared_ptr<Column<int> > col2 (new Column<int>("int column",INT));
    col->load("unittest_data/");

    if (!equals(reference_data, col)) {
      std::cerr << "STORE AND LOAD TEST FAILED!" << std::endl;
      return false;
    }
    std::cout << "SUCCESS" << std::endl;
  }

  return true;
}

bool unittest(boost::shared_ptr<ColumnBaseTyped<int> > col) {
  std::cout << "RUN Unittest for Column with BaseType ColumnBaseTyped<int> >"
            << std::endl;

  std::vector<int> reference_data(100);

  fill_column(col, reference_data);
  return test_column(col, reference_data);
}

bool unittest(boost::shared_ptr<ColumnBaseTyped<float> > col) {
  std::cout << "RUN Unittest for Column with BaseType ColumnBaseTyped<float> >"
            << std::endl;

  std::vector<float> reference_data(100);

  fill_column(col, reference_data);
  return test_column(col, reference_data);
}

bool unittest(boost::shared_ptr<ColumnBaseTyped<std::string> > col) {
  std::cout
      << "RUN Unittest for Column with BaseType ColumnBaseTyped<std::string> >"
      << std::endl;

  std::vector<std::string> reference_data(100);

  fill_column(col, reference_data);
  return test_column(col, reference_data);
}

bool GPU_accelerated_scans() {
  //            TableSchema schema;
  //            schema.push_back(Attribut(VARCHAR, "values"));
  //
  //            TablePtr tab(new Table("DictCompressedTable", schema));
  //            std::vector<std::string> reference_data;
  //
  //            for (unsigned int i = 0; i < 10; i++) {
  //                std::string val = get_rand_value<std::string>();
  //                reference_data.push_back(val);
  //            }
  //
  //            for (unsigned int i = 0; i < 1000; i++) {
  //
  //                {
  //                    Tuple t;
  //                    t.push_back(reference_data[rand() %
  //                    reference_data.size()]);
  //                    tab->insert(t);
  //                }
  //            }
  //
  //
  //            int selection_value;
  //            ValueComparator selection_comparison_value = EQUAL; //0 EQUAL, 1
  //            LESSER, 2 LARGER
  //
  //            boost::mt19937 rng;
  //            boost::uniform_int<> selection_values(0, reference_data.size() -
  //            1);
  //            //boost::uniform_int<> filter_condition(0,2);
  //
  //            hype::DeviceConstraint default_dev_constr =
  //            CoGaDB::RuntimeConfiguration::instance().getGlobalDeviceConstraint();
  //            for (unsigned int i = 0; i < 1000; i++) {
  //                selection_value = selection_values(rng);
  //
  //                KNF_Selection_Expression knf_expr; //(YEAR<2013 OR
  //                Name="Peter") AND (Age>18))
  //                {
  //                    Disjunction d;
  //                    d.push_back(Predicate("values",
  //                    boost::any(reference_data[selection_value]),
  //                    ValueConstantPredicate, EQUAL));
  //                    knf_expr.disjunctions.push_back(d);
  //                }
  //
  //                TablePtr cpu_result = BaseTable::selection(tab, "values",
  //                boost::any(reference_data[selection_value]), EQUAL, LOOKUP,
  //                SERIAL, CPU);
  //                TablePtr gpu_result = BaseTable::selection(tab, "values",
  //                boost::any(reference_data[selection_value]), EQUAL, LOOKUP,
  //                PARALLEL, GPU);
  //                //enforce CPU Use for Plan Generation
  //                CoGaDB::RuntimeConfiguration::instance().setGlobalDeviceConstraint(hype::CPU_ONLY);
  //                TablePtr cpu_complex_selection_result =
  //                BaseTable::selection(tab, knf_expr, LOOKUP);
  //                //enforce GPU Use for Plan Generation
  //                CoGaDB::RuntimeConfiguration::instance().setGlobalDeviceConstraint(hype::GPU_ONLY);
  //                TablePtr gpu_complex_selection_result =
  //                BaseTable::selection(tab, knf_expr, LOOKUP);
  //
  //                if
  //                (!cpu_result->getColumnbyName("values")->is_equal(gpu_result->getColumnbyName("values")))
  //                {
  //                    cerr << "Error in Simple GPU Selection! values Column
  //                    not correct!" << endl;
  //                    cout << "CPU Selection rows: " <<
  //                    cpu_result->getColumnbyName("values")->size() << " GPU
  //                    Selection rows: " <<
  //                    gpu_result->getColumnbyName("values")->size() << endl;
  //                    cout << "CPU Selection:" << endl;
  //                    cpu_result->getColumnbyName("values")->print();
  //                    cout << "GPU Selection:" << endl;
  //                    gpu_result->getColumnbyName("values")->print();
  //                    //            cout << "Nested Loop Join:" << endl;
  //                    //            result_nested_loop_join->print();
  //                    //            cout << "Hash Join:" << endl;
  //                    //            result_hash_join->print();
  //                    return false;
  //                }
  //
  //                if
  //                (!cpu_result->getColumnbyName("values")->is_equal(cpu_complex_selection_result->getColumnbyName("values")))
  //                {
  //                    cerr << "Error in Complex CPU Selection! values Column
  //                    not correct!" << endl;
  //                    cout << "CPU Selection rows: " <<
  //                    cpu_result->getColumnbyName("values")->size() << " CPU
  //                    Complex Selection rows: " <<
  //                    cpu_complex_selection_result->getColumnbyName("values")->size()
  //                    << endl;
  //                    return false;
  //                }
  //
  //                if
  //                (!cpu_result->getColumnbyName("values")->is_equal(gpu_complex_selection_result->getColumnbyName("values")))
  //                {
  //                    cerr << "Error in Complex GPU Selection! values Column
  //                    not correct!" << endl;
  //                    cout << "CPU Selection rows: " <<
  //                    cpu_result->getColumnbyName("values")->size() << " GPU
  //                    Selection rows: " <<
  //                    gpu_complex_selection_result->getColumnbyName("values")->size()
  //                    << endl;
  //                    return false;
  //                }
  //
  //                //col_string->selection()
  //            }
  //            //set Global Device Constraint to initial value
  //            CoGaDB::RuntimeConfiguration::instance().setGlobalDeviceConstraint(default_dev_constr);
  return true;
}

}  // end namespace unit_tests

}  // end namespace CoGaDB

int main(int argc, char* argv[]) {
  if (!CoGaDB::unit_tests::compressioned_columns_tests()) {
    return -1;
  }
  return 0;
}