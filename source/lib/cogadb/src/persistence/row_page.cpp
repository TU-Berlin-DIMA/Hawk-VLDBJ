#include <persistence/row_page.hpp>

using namespace std;

namespace CoGaDB {
RowPage::RowPage(unsigned short columns)
    : _rows(),
      _row_offsets(),
      _col_count(columns),
      _row_count(0),
      _is_full(false),
      _current_data_size(0) {
  _rows = new char[getMaxDataSize()];
  _row_offsets.push_back(0);
}

RowPage::RowPage(const RowPage& rp)
    : _rows(rp._rows),
      _row_offsets(rp._row_offsets),
      _col_count(rp._col_count),
      _row_count(rp._row_count),
      _is_full(rp._is_full),
      _current_data_size(rp._current_data_size) {}

RowPage& RowPage::operator=(const RowPage&) { return *this; }

bool RowPage::isFull() const { return _is_full; }

bool RowPage::fillPage(vector<BufferObject*> pages, int start) {
  // cout << "DEBUG: (RowPage) Fill row page with " << pages.size() << "
  // column(s) begin at " << start << endl;

  unsigned short* offset =
      new unsigned short(_row_offsets.back()); /* The offset for the next row */
  unsigned short* currentRowSize = new unsigned short(0); /* The size of the
                                                             created row in
                                                             bytes, including
                                                             bytes for column
                                                             size */

  int position = start; /* Set position to parameter start for adding only
                           column values after (+including) start  */
  char* row;            /* The current created row */

  /* Create new row for until the page is full */
  while (!isFull()) {
    row = createRow(pages, position, *offset,
                    *currentRowSize); /* Create row and get
                                         current row size and
                                         offset for next row */

    if (row == NULL) /* No row created (BufferPage holds no more values) ->
                        return false */
      return false;

    if ((_current_data_size + *currentRowSize) >=
        getMaxDataSize()) /* If (stored data + size of the
                             current row) are bigger then
                             the maximum of the page size
                             -> page is full */
    {
      _is_full = true;
      break;
    }

    position++; /* Increase the position for the next row to get next values
                   from columns (buffered pages) */

    _row_offsets.push_back(*offset); /* Stores the offset to the next row */

    /* Write row to page */
    for (int index = 0; index < *currentRowSize; index++) {
      _rows[_current_data_size++] = row[index];
    }

    /* Increase number of rows */
    _row_count++;
  }
  return true;
}

char* RowPage::createRow(vector<BufferObject*> pages, int pos,
                         unsigned short& offset, unsigned short& resRowSize) {
  int countOfColumns = pages.size();
  vector<char*> tmpValues;        /* Temporary stored values */
  unsigned short rowDataSize = 0; /* Size of this full row in bytes */
  unsigned short* tmpSize =
      new unsigned short(0);        /* Size of an value in bytes */
  vector<unsigned short> valueSize; /* Temporary stored size of value */

  /* Iterate throw all columns and temporary save value and size */
  for (int index = 0; index < countOfColumns; index++) {
    char* value = getValueFromPage(
        pages[index], pos, *tmpSize); /* Gets the value at position 'pos' of
                                         column 'index' and returns also the
                                         size of this value in bytes*/

    if (value == NULL) /* No more content on page */
      break;

    valueSize.push_back(*tmpSize); /* Temporary save the value size in bytes */
    tmpValues.push_back(value);    /* Temporary save the value */
    rowDataSize += *tmpSize; /* Increase the row size by value size - size for
                                column size will be saved later */
  }

  if (rowDataSize == 0) /* No data for this row -> abort */
    return NULL;

  unsigned short resultSizeInBytes =
      rowDataSize + 2 * countOfColumns; /* Row size + 2 bytes for each
                                           column to address value offset */
  char* res =
      new char[resultSizeInBytes]; /* Allocate memory for result array */
  char* valueOffset; /* 2 bytes for temporary stored offset of the value in this
                        row */

  /* Write all offsets to the beginning of the row - each offset contains 2
   * bytes */
  for (int index = 0; index < countOfColumns; index++) {
    valueOffset = reinterpret_cast<char*>(&valueSize[index]);
    res[(2 * index)] = valueOffset[0];
    res[(2 * index) + 1] = valueOffset[1];
  }

  unsigned short resIndex =
      2 * countOfColumns; /* Index of the value data in the result array */

  /* Write down all values for this row */
  for (int index = 0; index < countOfColumns; index++) {
    unsigned short vSize = valueSize[index]; /* Size of the current value -
                                                stored in separat variable for
                                                performance optimization */
    char* data = tmpValues[index]; /* Current data (value) - stored in separat
                                      variable for performance optimization */

    /* Write value down */
    for (unsigned short valueByteIndex = 0; valueByteIndex < vSize;
         valueByteIndex++) {
      res[resIndex++] = data[valueByteIndex];
    }
  }

  offset += resultSizeInBytes; /* Increase offset for next row by row size + 2
                                  bytes for each column  */
  resRowSize = resultSizeInBytes; /* Set row size in bytes (function parameter -
                                     additional return value) */

  return res;
}

char* RowPage::getValueFromPage(BufferObject* bo, int index,
                                unsigned short& valueSize) {
  AttributeType type =
      bo->getPage()
          ->getType(); /* Gets the attribute type of the current column */

  if (type == VARCHAR) /* Handle varchars in additional function */
    return getStringFromPage(bo, index, valueSize);

  int count = 0;
  /* Gets the right buffered object (linked list) based on the overall index
   * (index is zero based; count one based )*/
  while (true) {
    count = bo->getPage()->count();
    if (index <= count - 1) break;

    index -= bo->getPage()->count();
    bo = bo->next();
    if (bo == NULL) /* Index > stored values in this column */
      return NULL;
  }

  return bo->getPage()->getValueByIndex(index, valueSize);
}

char* RowPage::getStringFromPage(BufferObject* bo, int index,
                                 unsigned short& valueSize) {
  int count = 0;
  /* Gets the right buffered object (linked list) based on the overall index
   * (index is zero based; count one based )*/
  while (true) {
    count = bo->getPage()->count();
    if (index <= count - 1) break;

    index -= count;
    bo = bo->next();

    if (bo == NULL) /* Index > stored values in this column */
      return NULL;
  }
  /* Index is now the i-th value at the buffered page */

  return bo->getPage()->getStringByIndex(index, valueSize);
}

void RowPage::print(const TableSchema& schema) const {
  TableSchema::const_iterator it; /* Schema iterator */
  AttributeType type;
  int index = 0;
  int valueIndex = 0;

  unsigned short cols = schema.size();

  while (index < _row_count) {
    unsigned short startOffset = _row_offsets[index];
    int colOffset = 0;

    for (it = schema.begin(); it != schema.end(); it++) {
      type = it->first;
      char* size = new char[2];
      size[0] = _rows[startOffset + (2 * valueIndex)];
      size[1] = _rows[startOffset + (2 * valueIndex) + 1];
      valueIndex++;

      unsigned short valueSize = *reinterpret_cast<unsigned short*>(size);

      char* value = new char[valueSize];

      int i;
      bool b;
      float f;
      string s;

      for (int i = 0; i < valueSize; i++) {
        if (type == VARCHAR)
          s.push_back(_rows[startOffset + (2 * cols) + colOffset + i]);
        else
          value[i] = _rows[startOffset + (2 * cols) + colOffset + i];
      }

      colOffset += valueSize;

      switch (type) {
        case INT:
          i = *reinterpret_cast<int*>(value);
          cout << i << "\t";
          break;
        case BOOLEAN:
          b = *reinterpret_cast<bool*>(value);
          cout << b << "\t";
          break;

        case VARCHAR:
          cout << s << "\t";
          break;
        case FLOAT:
          f = *reinterpret_cast<float*>(value);
          cout << f << "\t";
          break;
        default:
          break;
      }
    }
    cout << endl;

    valueIndex = 0;

    index++;
  }

  cout << "Rows: " << _row_count << endl;
  cout << "Cols: " << cols << endl;
  cout << "Data: " << _current_data_size << endl;
}

unsigned short RowPage::getMaxDataSize() const {
  // char bytes = MAX_SIZE_BYTES;
  return ((1 << (16)) - 1);
}

unsigned short RowPage::getDataSize() const { return _current_data_size; }

unsigned short RowPage::count() const { return _row_count; }

char* RowPage::getValue(unsigned int page_row, unsigned int col,
                        unsigned short& length) const {
  unsigned short offset = _row_offsets[page_row];
  unsigned short size = 0;
  unsigned int col_offset = 0;
  char* tmpSize = new char[2];
  char* res;

  for (unsigned int index = 0; index < col - 1; index++) {
    tmpSize[0] = _rows[offset + 2 * index];
    tmpSize[1] = _rows[offset + 2 * index + 1];

    size = *reinterpret_cast<unsigned short*>(tmpSize);
    col_offset += size;
  }

  if (size == 0) {
    tmpSize[0] = _rows[offset];
    tmpSize[1] = _rows[offset + 1];
    size = *reinterpret_cast<unsigned short*>(tmpSize);
  }

  col_offset += 2 * _col_count;

  res = new char[size];

  for (unsigned short index = 0; index < size; index++) {
    res[index] = _rows[offset + col_offset + index];
  }

  length = size;

  return res;
}

char* RowPage::getData() const { return _rows; }
}