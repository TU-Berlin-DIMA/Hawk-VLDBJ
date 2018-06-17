#pragma once
#include <core/column.hpp>
#include <core/column_base_typed.hpp>
#include <core/row_table.hpp>

using namespace std;
namespace CoGaDB {
  /*! \brief	Interface for column based access to the early materialized
   * table RowTable */

  template <typename T>
  class RowValueColumn : public ColumnBaseTyped<T> {
   private:
    /*! \brief Pointer to RowTable (early materialized table) to which this
     * Column depents */
    RowTablePtr _row_table_ptr;
    /*! \brief Column Index of the RowTable */
    unsigned int _column_index;
    /*! \brief Type of this column */
    AttributeType _type;
    /*! \brief Gets a value at a specific row (index) as bytes (char*) */
    char* getValue(const int index, unsigned short& stringLength) const;

   public:
    RowValueColumn(RowTablePtr, string&, const AttributeType&);
    virtual ~RowValueColumn();

    bool isCompressed() const throw();
    bool update(PositionListPtr tid, const boost::any& new_value);
    bool remove(PositionListPtr tid);

    /*! \brief - Not implemented - */
    bool insert(const boost::any& new_Value);
    /*! \brief - Not implemented - */
    bool update(TID tid, const boost::any& new_Value);
    /*! \brief - Not implemented - */
    bool remove(TID tid);

    virtual const boost::any get(TID tid);
    virtual void print() const throw();
    /*! \brief - Not implemented - */
    size_t size() const throw();
    /*! \brief - Not implemented - */
    unsigned int getSizeinBytes() const throw();

    /*! \brief - Not implemented - */
    const ColumnPtr copy() const;

    /*! \brief - Not implemented - */
    bool store(const string& path);
    /*! \brief - Not implemented - */
    bool load(const string& path);
    /*! \brief Always false */
    bool isMaterialized() const throw();
    /*! \brief returns type information of internal values */
    const type_info& type() const throw();

    /*! \brief Index-based access. Use for accessing values in column */
    T& operator[](const int index);

    ColumnPtr toColumn();
  };

  template <typename T>
  RowValueColumn<T>::RowValueColumn(RowTablePtr row_table_ptr, string& name,
                                    const AttributeType& type)
      : ColumnBaseTyped<T>(name, type),
        _row_table_ptr(),
        _column_index(0),
        _type() {
    _row_table_ptr = row_table_ptr;
    TableSchema schema = _row_table_ptr->getSchema();
    TableSchema::const_iterator it; /* Schema iterator */

    /* Iterate throw all columns of the schema and get the relative pages from
     * BufferManager */
    for (it = schema.begin(); it != schema.end(); it++) {
      _column_index++;
      if (it->second == name) {
        _type = it->first;
        break;
      }
    }
  }

  template <typename T>
  RowValueColumn<T>::~RowValueColumn() {}

  template <typename T>
  bool RowValueColumn<T>::insert(const boost::any& new_Value) {
    if (new_Value.empty()) return false;

    return false;
  }

  template <typename T>
  bool RowValueColumn<T>::update(TID tid, const boost::any& new_Value) {
    cout << tid << new_Value.empty() << endl;
    return false;
  }

  template <typename T>
  bool RowValueColumn<T>::remove(TID tid) {
    cout << tid;
    return false;
  }

  template <typename T>
  const boost::any RowValueColumn<T>::get(TID tid) {
    unsigned short* stringLength = new unsigned short(0);

    char* res = getValue(tid, *stringLength);
    string s;
    int i;
    float f;
    bool b;

    switch (_type) {
      case INT:
        i = *reinterpret_cast<int*>(res);
        return boost::any(i);
      case FLOAT:
        f = *reinterpret_cast<float*>(res);
        return boost::any(f);
      case BOOLEAN:
        b = *reinterpret_cast<bool*>(res);
        return boost::any(b);
      case VARCHAR:
        for (unsigned short index = 0; index < *stringLength; index++)
          s.push_back(res[index]);

        return boost::any(s);
    }

    return boost::any();
  }

  template <typename T>
  void RowValueColumn<T>::print() const throw() {
    unsigned short* stringLength = new unsigned short(0);
    for (size_t index = 0; index < _row_table_ptr->getNumberofRows(); index++) {
      char* res = getValue(index, *stringLength);
      string s;

      switch (_type) {
        case INT:
          cout << *reinterpret_cast<int*>(res) << endl;
          break;
        case FLOAT:
          cout << *reinterpret_cast<float*>(res) << endl;
          break;
        case BOOLEAN:
          cout << *reinterpret_cast<bool*>(res) << endl;
          break;
        case VARCHAR:
          for (unsigned short index = 0; index < *stringLength; index++)
            s.push_back(res[index]);

          cout << s << endl;
          break;
      }
    }
  }

  template <typename T>
  size_t RowValueColumn<T>::size() const throw() {
    /* Not implemented */
    return 0;
  }

  template <typename T>
  unsigned int RowValueColumn<T>::getSizeinBytes() const throw() {
    /* Not implemented */
    return 0;
  }

  template <typename T>
  const ColumnPtr RowValueColumn<T>::copy() const {
    /* Not implemented */
    return ColumnPtr();
  }

  template <typename T>
  bool RowValueColumn<T>::store(const string& path) {
    /* Not implemented */
    cout << path << endl;
    return false;
  }

  template <typename T>
  bool RowValueColumn<T>::load(const string& path) {
    /* Not implemented */
    cout << path << endl;
    return false;
  }

  template <typename T>
  bool RowValueColumn<T>::isMaterialized() const throw() {
    /* Always false */
    return false;
  }

  template <typename T>
  const std::type_info& RowValueColumn<T>::type() const throw() {
    return typeid(T);
  }

  template <>
  inline int& RowValueColumn<int>::operator[](const int index) {
    unsigned short* stringLength = new unsigned short(0);

    char* res = getValue(index, *stringLength);

    return *reinterpret_cast<int*>(res);
  }

  template <>
  inline string& RowValueColumn<string>::operator[](const int index) {
    unsigned short* stringLength = new unsigned short(0);

    char* res = getValue(index, *stringLength);
    string* s = new string();

    for (unsigned short index = 0; index < *stringLength; index++)
      s->push_back(res[index]);

    return *s;
  }

  template <>
  inline bool& RowValueColumn<bool>::operator[](const int index) {
    unsigned short* stringLength = new unsigned short(0);

    char* res = getValue(index, *stringLength);

    return *reinterpret_cast<bool*>(res);
  }

  template <>
  inline float& RowValueColumn<float>::operator[](const int index) {
    unsigned short* stringLength = new unsigned short(0);

    char* res = getValue(index, *stringLength);

    return *reinterpret_cast<float*>(res);
  }

  template <typename T>
  char* RowValueColumn<T>::getValue(const int index,
                                    unsigned short& stringLength) const {
    unsigned int* prefix = new unsigned int(0);
    RowPagePtr rpp = _row_table_ptr->getPageByIndex(index, *prefix);

    /* Call getValue() from RowPage */
    return rpp->getValue(index - (*prefix), _column_index, stringLength);
  }

  template <typename T>
  bool RowValueColumn<T>::isCompressed() const throw() {
    return false;
  }

  template <typename T>
  bool RowValueColumn<T>::update(PositionListPtr tid,
                                 const boost::any& new_value) {
    cout << tid << boost::any_cast<string>(new_value);
    return false;
  }

  template <typename T>
  bool RowValueColumn<T>::remove(PositionListPtr tid) {
    cout << tid;
    return false;
  }

  template <typename T>
  ColumnPtr RowValueColumn<T>::toColumn() {
    string name = this->getName();
    ColumnPtr col = createColumn(this->getType(), this->getName());
    size_t count = _row_table_ptr->getNumberofRows();

    for (size_t index = 0; index < count; index++) {
      col->insert((this)->operator[](index));
    }

    return col;
  }
}