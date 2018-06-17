#pragma once
#include <climits>
#include <core/global_definitions.hpp>
#include <persistence/buffer_object.hpp>
#include <set>

namespace CoGaDB {

#define MAX_PAGES UINT_MAX

  typedef shared_pointer_namespace::shared_ptr<BufferObject> BuffObjPtr;

  /* BufferManager implementet as Singleton Pattern */
  class BufferManager {
   private:
    static BufferManager* _instance;
    unsigned int _maxSize;
    /*! B+ tree structure for fast access */
    std::set<BuffObjPtr> _pages;
    /*! \brief Private constructor for singleton pattern */
    BufferManager();
    /*! \brief Load page from HDD */
    BufferObject* createPage(std::string, std::string, int);

   public:
    /*! \brief Get instance of BufferManager -> singleton pattern  */
    static BufferManager* getInstance();
    bool pageReplacement();
    int count() const;
    void print() const;
    BufferObject* getPages(std::string, std::string);
  };
}