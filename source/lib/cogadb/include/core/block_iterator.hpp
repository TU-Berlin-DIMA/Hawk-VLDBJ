/*
 * File:   block_iterator.hpp
 * Author: sebastian
 *
 * Created on 27. Dezember 2015, 16:05
 */

#ifndef BLOCK_ITERATOR_HPP
#define BLOCK_ITERATOR_HPP

#include <stddef.h>
#include <boost/shared_ptr.hpp>

namespace CoGaDB {
  class BlockIterator;
  typedef boost::shared_ptr<BlockIterator> BlockIteratorPtr;

  class BlockIterator {
   public:
    BlockIterator();
    BlockIterator(size_t block_size);
    bool hasNext() const noexcept;
    void advance() noexcept;
    void reset() noexcept;
    size_t getOffset() const noexcept;
    size_t getBlockSize() const noexcept;

   private:
    size_t offset;
    size_t num_elements_per_block;
    size_t has_next;
  };

}  // end namespace CoGaDB

#endif /* BLOCK_ITERATOR_HPP */
