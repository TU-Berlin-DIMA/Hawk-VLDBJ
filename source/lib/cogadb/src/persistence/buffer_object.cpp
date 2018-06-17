#include <persistence/buffer_object.hpp>

using namespace std;

namespace CoGaDB {
BufferObject::BufferObject(PagePtr page, string path, string table, string col)
    : _next(NULL),
      _prev(NULL),
      _page(page),
      _path(path),
      _table(table),
      _col(col) {}

BufferObject::~BufferObject() {
  // free(&_path);
  // free(&_page);
}

BufferObject::BufferObject(const BufferObject& bo)
    : _next(bo._next),
      _prev(bo._prev),
      _page(bo._page),
      _path(bo._path),
      _table(bo._table),
      _col(bo._col) {}

BufferObject* BufferObject::next() const { return _next; }

BufferObject* BufferObject::prev() const { return _prev; }

void BufferObject::setNext(BufferObject* bo) { _next = bo; }

void BufferObject::setPrev(BufferObject* bo) { _prev = bo; }

string BufferObject::getTableName() const { return _table; }

string BufferObject::getColName() const { return _col; }

PagePtr BufferObject::getPage() { return _page; }

string BufferObject::getPath() const { return _path; }

bool BufferObject::containsContentOf(string table, string col) const {
  /* It's the same content, if the table and column name are equal. */
  return (_table == table && _col == col);
}

void BufferObject::printStatus() const {
  _page->printStatus();

  if (_next != NULL) {
    cout << "--------------------" << endl;
    _next->printStatus();
  }
}
}