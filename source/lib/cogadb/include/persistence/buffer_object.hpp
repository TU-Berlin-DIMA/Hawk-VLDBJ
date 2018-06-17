#pragma once
#include <core/global_definitions.hpp>
#include <persistence/page.hpp>

namespace CoGaDB {

  /*! \brief Abstract object with holds a Page and more information of it. Holds
   * also Pointer to previous or next pages, which holds also data of the same
   * column.
   * This double linked list will be created by BufferManager.
   * Iterate with prev() and next().
   * For debugging, printStatus() prints the page-status-byte of this and all
   * next pages. */
  struct BufferObject {
   public:
    BufferObject(PagePtr, std::string, std::string, std::string);
    BufferObject(const BufferObject&);
    ~BufferObject();
    BufferObject& operator=(const BufferObject&);
    PagePtr getPage();
    std::string getPath() const;
    // bool operator<(const BufferObject&) const;
    std::string getTableName() const;
    std::string getColName() const;
    BufferObject* next() const;
    BufferObject* prev() const;
    void setNext(BufferObject*);
    void setPrev(BufferObject*);
    bool containsContentOf(std::string, std::string) const;
    void printStatus() const;

   private:
    struct BufferObject* _next;
    struct BufferObject* _prev;
    PagePtr _page;
    std::string _path;
    std::string _table;
    std::string _col;
  };
}