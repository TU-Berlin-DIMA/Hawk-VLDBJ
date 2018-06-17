#include <persistence/buffer_manager.hpp>

using namespace std;

namespace CoGaDB {

BufferManager* BufferManager::_instance = NULL;

BufferManager::BufferManager() : _maxSize(), _pages() {
  _maxSize = _pages.max_size() < MAX_PAGES ? _pages.max_size() : MAX_PAGES;

  cout << "DEBUG: (BufferManager) Initializing." << endl;
  cout << "DEBUG: (BufferManager) Max Pages: " << _maxSize << endl;
}

BufferManager* BufferManager::getInstance() {
  if (_instance == NULL) {
    _instance = new BufferManager();
  }

  return _instance;
}

BufferObject* BufferManager::createPage(string table, string col, int offset) {
  string path("./bin/data/" + table + "/" + col + ".bin");
  cout << "DEBUG: (BufferManager) Loading " << path << " at offset " << offset
       << endl;
  char status;
  BufferObject* res = NULL;

  if (_pages.size() > _maxSize) {
    if (!pageReplacement()) return NULL;
  }

  ifstream source(path.c_str(), std::ios::in | std::ios::binary);

  source.read(&status, 1); /* Read status from file (first byte) */
  source.seekg(offset > 0 ? offset
                          : 1); /* set streamposition at beginning of content
                                   (>=1 because of status byte) */

  bool eof = source.eof();
  BufferObject* last = NULL;

  while (!eof) {
    PagePtr page = PagePtr(new Page(source, status));
    BuffObjPtr bo = BuffObjPtr(new BufferObject(page, path, table, col));

    if (last != NULL) {
      /* double linked list for reading a full column throw more then one page.
       */
      bo->setPrev(last);
      last->setNext(&(*bo));
    }

    if (res == NULL) {
      res = &(*bo);
    }

    last = &(*bo);

    // cout << "DEBUG: (BufferManager) Adding page for column '" << col << "' in
    // table '" << table << "'" << endl;

    _pages.insert(bo);

    eof = source.eof();
  }

  source.close();

  return res;
}

BufferObject* BufferManager::getPages(string table, string col) {
  // cout << "DEBUG: (BufferManager) Searching for pages of column '" << col <<
  // "' in table '" << table << "'" << endl;
  for (set<BuffObjPtr>::iterator it = _pages.begin(); it != _pages.end();
       ++it) {
    if ((*it)->containsContentOf(table, col)) {
      BufferObject* bo = &(*(*it));
      while (bo->prev() != NULL) {
        bo = bo->prev();
      }

      return bo;
    }
  }

  return createPage(table, col, 0);
}

bool BufferManager::pageReplacement() {
  /*! TODO: Add implementation */
  cout << "DEBUG: (BufferManager) Page Replacement - NOT IMPLEMENTED" << endl;
  return true;
}

int BufferManager::count() const { return _pages.size(); }

void BufferManager::print() const {
  for (set<BuffObjPtr>::iterator it = _pages.begin(); it != _pages.end();
       ++it) {
    (*it)->getPage()->print();
  }
}
}