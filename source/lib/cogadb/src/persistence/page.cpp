#include <persistence/page.hpp>

using namespace std;

namespace CoGaDB {
Page::Page(bool compressed, AttributeType type, bool bigDataSize)
    : _header(), _data(NULL), _isFull(false), _value_offsets() {
  char t = BIT_TYPE_LOW;  // Shift length for type
  char l = BIT_LENGTH;    // Shift length for max data size coeffizient

  _header.Status = (char)compressed;    // set compressed bit
  _header.Status |= (char)(type << t);  // set Type bits [from 00 to 11]
  _header.Status |=
      (char)(bigDataSize << l);  // set max data size coeffizient bit

  _header.Size = 0;
  _header.Count = 0;

  _data = new char[getMaxDataSize()];

  _header.Offset = DATA_OFFSET;
}

Page::Page(ifstream& source, char status)
    : _header(), _data(NULL), _isFull(false), _value_offsets() {
  _header.Status = status;
  _header.Offset = source.tellg();
  _header.Size = 0;
  _header.Count = 0;
  _data = new char[getMaxDataSize()];

  appendData(source);
}

Page::Page(const Page& p)
    : _header(p._header),
      _data(p._data),
      _isFull(p._isFull),
      _value_offsets(p._value_offsets) {}

Page& Page::operator=(const Page&) { return *this; }

Page::~Page() {
  // delete[] _data;
  // delete &_header;
}

bool Page::isCompressed() const {
  char pos = BIT_COMPRESSED;

  return (getStatusBitAt(pos) > 0);
}

AttributeType Page::getType() const {
  char pos_low = BIT_TYPE_LOW;
  char pos_high = BIT_TYPE_HIGH;

  char t = getStatusBitAt(pos_low);
  t |= getStatusBitAt(pos_high) << 1;

  AttributeType at = (AttributeType)t;

  return at;
}

bool Page::isDirty() const {
  char pos = BIT_DIRTY;

  return (getStatusBitAt(pos) > 0);
}

bool Page::isInUsage() const {
  char pos = BIT_USAGE;

  return (getStatusBitAt(pos) > 0);
}

unsigned short Page::getDataSize() const { return _header.Size; }

int Page::getFileOffset() const { return _header.Offset; }

unsigned short Page::getMaxDataSize() const {
  char pos = BIT_LENGTH;

  char l = getStatusBitAt(pos);
  return (1 << (l + 1) * 8) - 1;  // 2^([l+1]*8) => 2^8 or 2^16
}

void Page::printStatus() const {
  cout << "Compressed: " << this->isCompressed() << endl;
  cout << "Type: " << this->getType() << endl;
  cout << "Dirty: " << this->isDirty() << endl;
  cout << "Usage: " << this->isInUsage() << endl;
  cout << "MaxDataSize: " << this->getMaxDataSize() << " byte" << endl;
  cout << "Current DataSize: " << this->getDataSize() << " byte" << endl;
  cout << "Offset in File: " << _header.Offset << endl;
  cout << "Size of one value: " << this->getSizeOfOneValue() << " byte" << endl;
  cout << "Value count: " << this->count() << endl;
}

char Page::getStatusBitAt(char pos) const {
  char b = 1 << pos;
  return (b & this->_header.Status) >> pos;
}

bool Page::appendData(ifstream& source) {
  // char pos = BIT_DIRTY;
  char t = getType();
  bool res;

  switch (t) {
    case INT:
    case FLOAT:
    case BOOLEAN:
      res = append(source);
      break;
    case VARCHAR:
      res = appendVarchar(source);
      break;

    default:
      res = false;  // not able to add data
  }

  /* Set Dirty Bit */
  // setStatusBit(pos, 1);

  return res;
}

bool Page::appendVarchar(ifstream& source) {
  int stream_pos;
  unsigned short str_len;
  unsigned short max = getMaxDataSize();
  int shift = VARCHAR_LENGTH_BYTES;
  char* tmp = new char[(1 << (shift * 8))];
  char* len = new char[shift];

  try {
    while (source.read(len, shift)) {
      _value_offsets.push_back(_header.Size);
      str_len = *reinterpret_cast<unsigned short*>(len);

      if (_header.Size > max - (str_len + shift)) {
        /* seek back */
        stream_pos = source.tellg();
        stream_pos -= shift;

        source.seekg(stream_pos);

        _isFull = true;

        return true;
      }

      source.read(tmp, str_len);

      for (int index = 0; index < shift; index++)
        _data[_header.Size++] = len[index];

      for (int index = 0; index < str_len; index++) {
        _data[_header.Size++] = tmp[index];
      }

      _header.Count++;
    }
  } catch (...) {
    cout << "Exception occured in Page::appendVarchar(ifstream&) while reading "
            "from file or writing to page."
         << endl;
    return false;
  }

  return true;
}

bool Page::append(ifstream& source) {
  int s = getSizeOfOneValue();

  unsigned short max = getMaxDataSize();
  char* tmp = new char[s];
  try {
    /* Enough space on page for at leased one value? */
    if (_header.Size > max - s) {
      return true;
    }

    while (source.read(tmp, s)) {
      _value_offsets.push_back(_header.Size);
      for (int index = 0; index < s; index++)
        _data[_header.Size++] = tmp[index];

      _header.Count++;

      /* Enough space on page for another value? */
      if (_header.Size > max - s) {
        return true;
      }
    }

  } catch (...) {
    cout << "Exception occured in Page::append(ifstream&) while reading from "
            "file or writing to page."
         << endl;
    return false;
  }

  return true;
}

int Page::getSizeOfOneValue() const {
  char t = getType();
  switch (t) {
    case INT:
    case FLOAT:
      return 4;
    case BOOLEAN:
      return 1;

    default:
      return -1;
      break;
  }
}

void Page::setStatusBit(char pos, bool val) {
  char b = val << pos;
  _header.Status = (b | this->_header.Status);
}

void Page::print() const {
  AttributeType type = getType();
  int size = getSizeOfOneValue();
  unsigned short index = 0;
  while (index < _header.Size) {
    switch (type) {
      case INT:
        printInt(index);
        break;
      case FLOAT:
        printFloat(index);
        break;
      case BOOLEAN:
        printBool(index);
        break;
      case VARCHAR:
        size = printVarchar(index);
        break;

      default:
        return;
    }

    index += size;
  }
}

void Page::printInt(unsigned short pos) const {
  char* tmp = new char[4];

  tmp[0] = _data[pos];
  tmp[1] = _data[pos + 1];
  tmp[2] = _data[pos + 2];
  tmp[3] = _data[pos + 3];

  cout << *reinterpret_cast<int*>(tmp) << endl;
}

void Page::printFloat(unsigned short pos) const {
  char* tmp = new char[4];

  tmp[0] = _data[pos];
  tmp[1] = _data[pos + 1];
  tmp[2] = _data[pos + 2];
  tmp[3] = _data[pos + 3];

  cout << *reinterpret_cast<float*>(tmp) << endl;
}

void Page::printBool(unsigned short pos) const {
  char* tmp = new char[1];
  tmp[0] = _data[pos];

  cout << *reinterpret_cast<bool*>(tmp) << endl;
}

int Page::printVarchar(unsigned short pos) const {
  unsigned short len;
  int bytes = VARCHAR_LENGTH_BYTES;
  char* str_len = new char[bytes];

  for (int index = 0; index < bytes; index++)
    str_len[index] = _data[pos + index];

  len = *reinterpret_cast<unsigned short*>(str_len);

  for (int index = 0; index < len; index++) cout << _data[pos + bytes + index];

  cout << endl;

  return len + bytes;
}

bool Page::isFull() const {
  int one = getSizeOfOneValue();
  int max = getMaxDataSize();
  int size = getDataSize();

  one = one > 0 ? one : 3;  // 3 bytes, because of 2 bytes for string length and
                            // at leased one byte for one char

  return _isFull || (size > max - one);
}

char* Page::getData() { return _data; }

unsigned short Page::count() const { return _header.Count; }

char* Page::getStringByIndex(unsigned int index, unsigned short& valueSize) {
  char* str_len = new char[2]; /* Temporary storage for string length */
  unsigned short size;
  unsigned int start = _value_offsets[index];
  /* Get the length of the very right string */
  str_len[0] = _data[start];
  str_len[1] = _data[start + 1];

  size = *reinterpret_cast<unsigned short*>(str_len);
  start +=
      2; /* Increase start offset for string value by 2 bytes (length bytes) */

  char* result = new char[size]; /* String storage */

  /* Write value */
  for (int i = 0; i < size; i++) {
    result[i] = _data[(start) + i];
  }

  valueSize = size; /* Set size of this value - additional return value */

  return result;
}

char* Page::getValueByIndex(unsigned int index, unsigned short& valueSize) {
  unsigned short size = getSizeOfOneValue();
  char* result = new char[size]; /* Storage for value */
  int start = _value_offsets[index];

  /* Read value */
  for (int i = 0; i < size; i++) result[i] = _data[(start) + i];

  valueSize = size; /* Set size of this value - additional return value */

  return result;
}
}
