#include <core/row_table.hpp>
#include <core/row_value_column.hpp>

using namespace std;

namespace CoGaDB {

RowTable::RowTable(const string& name, const TableSchema& schema)
    : BaseTable(name, schema),
      _rtp(),
      _row_pages(),
      _row_pages_prefix(),
      _row_page_mapping(),
      _cols(),
      _row_count(0),
      _isMaterialized(true) {}

RowTable::~RowTable() {
  // free(&_cols);
  // free(&_isMaterialized);
}

bool RowTable::store(const std::string& path_to_table_dir) { return false; }

bool RowTable::load(TableLoaderMode loader_mode) {
  _rtp = RowTablePtr(this);

  cout << "DEBUG: (RowTable) Loading..." << endl;

  TableSchema::const_iterator it; /* Schema iterator */
  TableSchema schema = getSchema();
  int nmbrOfCols = schema.size();
  BufferManager* bm = BufferManager::getInstance();
  string col;
  vector<BufferObject*> bos;

  cout << "DEBUG: (RowTable) Creating table '" << name_ << "' with "
       << nmbrOfCols << " column(s)" << endl;

  /* Iterate throw all columns of the schema and get the relative pages from
   * BufferManager */
  for (it = schema.begin(); it != schema.end(); it++) {
    col = it->second;
    cout << "DEBUG: (RowTable) Adding column '" << col << "' to table '"
         << name_ << "'" << endl;

    /* Get pages from BufferManager */
    BufferObject* bo = bm->getPages(name_, col);
    bos.push_back(bo);
    AttributeType type = it->first;

    /* add specific RowValueColumn to list of columns */
    if (type == INT) {
      _cols.push_back(ColumnPtr(new RowValueColumn<int>(_rtp, col, it->first)));
    } else if (type == BOOLEAN) {
      _cols.push_back(
          ColumnPtr(new RowValueColumn<bool>(_rtp, col, it->first)));
    } else if (type == FLOAT) {
      _cols.push_back(
          ColumnPtr(new RowValueColumn<float>(_rtp, col, it->first)));
    } else if (type == VARCHAR) {
      _cols.push_back(
          ColumnPtr(new RowValueColumn<string>(_rtp, col, it->first)));
    }
  }

  cout << "DEBUG: (RowTable) Materialize table '" << name_ << "' ... " << endl;

  /* Materialize - fill RowPage's until no more data are available <fillPage ==
   * false> (materialize all data [early materialization]) */
  RowPage* rp = new RowPage(nmbrOfCols);

  Timestamp start;
  Timestamp end;
  start = getTimestamp();

  while (rp->fillPage(bos, _row_count)) {
    _row_count += rp->count();

    /* Store row page */
    _row_pages.push_back(RowPagePtr(rp));
    updateMapping(rp->count(), _row_pages.size() - 1);
    rp = new RowPage(nmbrOfCols);
  }

  /* fillPage result with false (no more data are available) but this row page
   * was not stored */
  if (rp->count() > 0) {
    _row_count += rp->count();
    _row_pages.push_back(RowPagePtr(rp));
    updateMapping(rp->count(), _row_pages.size() - 1);
  }

  end = getTimestamp();
  cout << "(MEASUREMENT) Materialize: "
       << ((double)end - (double)start) / 1000000 << " ms" << endl;

  unsigned int rps = _row_pages.size();
  cout << "DEBUG: (RowOrientedTable) " << rps << " row pages added." << endl;

  return true;
}

bool RowTable::loadDatafromFile(string filepath) {
  /* Not implmented. Use load() */
  cout << "DEBUG: (RowTable) Loading from " << filepath << endl;
  return false;
}

const TablePtr RowTable::materialize() const {
  /* Not implemented */
  return TablePtr();
}

size_t RowTable::getNumberofRows() const throw() { return _row_count; }

bool RowTable::isMaterialized() const throw() {
  /* Alwys true, becuase of early materialization at loading time */
  return _isMaterialized;
}

const Tuple RowTable::fetchTuple(const TID& id) const {
  cout << "DEBUG: (RowTable) Fetching tuple " << id << endl;
  const Tuple t;

  return t;
}

bool RowTable::insert(const Tuple& t) {
  /* Not implemented */
  cout << "DEBUG: (RowTable) Insert tuple with " << t.size() << " values"
       << endl;
  return false;
}

bool RowTable::update(const string& attribute_name, const boost::any& value) {
  /* Not implemented */
  cout << "DEBUG: (RowTable) Update " << attribute_name
       << " (is empty: " << value.empty() << ")" << endl;
  return false;
}

bool RowTable::remove(const string& attribute_name, const boost::any& value) {
  /* Not implemented */
  cout << "DEBUG: (RowTable) Remove " << attribute_name
       << " (is empty: " << value.empty() << ")" << endl;
  return false;
}

const ColumnPtr RowTable::getColumnbyName(const string& column_name) const
    throw() {
  /* Iterate throw all columns */
  unsigned short size = _cols.size();
  for (unsigned int index = 0; index < size; index++) {
    /* if name match to column_name - return column */
    if (_cols[index]->getName() == column_name) return _cols[index];
  }

  /* No column found */
  return ColumnPtr();
}

const vector<ColumnPtr>& RowTable::getColumns() const { return _cols; }

void RowTable::print() {
  TableSchema::const_iterator it; /* Schema iterator */
  TableSchema schema = getSchema();
  unsigned int colIndex = 0;
  unsigned int rowIndex = 0;
  AttributeType type;

  while (rowIndex < _row_count) {
    for (it = schema.begin(); it != schema.end(); it++) {
      type = it->first;
      ColumnPtr cPtr = getColumnbyName(it->second);
      // value = cPtr->get(rowIndex);

      if (type == INT) {
        shared_pointer_namespace::shared_ptr<ColumnBaseTyped<int> > column =
            shared_pointer_namespace::static_pointer_cast<
                ColumnBaseTyped<int> >(cPtr);
        cout << (*column)[rowIndex] << "\t";
      }

      if (type == FLOAT) {
        shared_pointer_namespace::shared_ptr<ColumnBaseTyped<float> > column =
            shared_pointer_namespace::static_pointer_cast<
                ColumnBaseTyped<float> >(cPtr);
        cout << (*column)[rowIndex] << "\t";
      }

      if (type == BOOLEAN) {
        shared_pointer_namespace::shared_ptr<ColumnBaseTyped<bool> > column =
            shared_pointer_namespace::static_pointer_cast<
                ColumnBaseTyped<bool> >(cPtr);
        cout << (*column)[rowIndex] << "\t";
      }

      if (type == VARCHAR) {
        shared_pointer_namespace::shared_ptr<ColumnBaseTyped<string> > column =
            shared_pointer_namespace::static_pointer_cast<
                ColumnBaseTyped<string> >(cPtr);
        cout << (*column)[rowIndex] << "\t";
      }

      colIndex++;
    }

    cout << endl;
    rowIndex++;
  }
}

const std::vector<RowPagePtr>& RowTable::getRowPages() const {
  return _row_pages;
}

bool RowTable::update(PositionListPtr tids, const std::string& attribute_name,
                      const boost::any& value) {
  cout << tids << attribute_name << boost::any_cast<string>(value);
  return false;
}

bool RowTable::remove(PositionListPtr tids) {
  cout << tids;
  return false;
}

RowPagePtr RowTable::getPageByIndex(unsigned int index, unsigned int& prefix) {
  /*int rp_index = 0;
  while(index > _row_pages_prefix[rp_index])
  {
          prefix = _row_pages_prefix[rp_index];
          rp_index++;
  }
  */

  prefix =
      _row_pages_prefix[index] - _row_pages[_row_page_mapping[index]]->count();

  // cout << prefix << endl;
  // cout << _row_pages[_row_page_mapping[index]] << endl;

  return _row_pages[_row_page_mapping[index]];
}

void RowTable::updateMapping(unsigned int count, unsigned int value) {
  for (unsigned int index = 0; index < count; index++) {
    _row_page_mapping.push_back(value);
    _row_pages_prefix.push_back(_row_count);
  }
}
}