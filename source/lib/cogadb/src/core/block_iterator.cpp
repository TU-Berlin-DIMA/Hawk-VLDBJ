
#include <core/block_iterator.hpp>

namespace CoGaDB {

BlockIterator::BlockIterator()
    : offset(0), num_elements_per_block(0), has_next(true) {}

BlockIterator::BlockIterator(size_t _block_size)
    : offset(0), num_elements_per_block(_block_size), has_next(true) {}

bool BlockIterator::hasNext() const noexcept { return has_next; }

void BlockIterator::advance() noexcept {
  this->offset += this->num_elements_per_block;

  //            if(this->offset+this->num_elements_per_block>this->total_num_elements){
  //                this->offset+=this->num_elements_per_block;
  //            }else{
  //                //reached end of relation
  //                this->offset=this->total_num_elements;
  //            }
}

void BlockIterator::reset() noexcept {
  this->offset = 0;
  this->has_next = true;
}

size_t BlockIterator::getOffset() const noexcept { return this->offset; }

size_t BlockIterator::getBlockSize() const noexcept {
  return this->num_elements_per_block;
}
}  // end namespace CoGaDB
