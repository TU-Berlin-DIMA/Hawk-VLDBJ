#pragma once

#include <boost/shared_ptr.hpp>
#include <core/global_definitions.hpp>
#include <core/memory_allocator.hpp>

namespace CoGaDB {

  class Bitmap {
   public:
    typedef
        typename MemoryAllocator<char>::MemoryAllocatorPtr MemoryAllocatorPtr;
    typedef boost::shared_ptr<Bitmap> BitmapPtr;
    Bitmap(char* bitmap, size_t num_of_bits,
           const hype::ProcessingDeviceMemoryID& mem_id = hype::PD_Memory_0);
    Bitmap(size_t num_of_bits, bool init_value = false, bool initialize = true,
           const hype::ProcessingDeviceMemoryID& mem_id = hype::PD_Memory_0);
    ~Bitmap();
    char* data();
    size_t size() const;
    hype::ProcessingDeviceMemoryID getMemoryID() const;
    BitmapPtr copy() const;
    BitmapPtr copy(const hype::ProcessingDeviceMemoryID&) const;

   private:
    Bitmap(const Bitmap&);
    Bitmap& operator=(const Bitmap&);
    char* bitmap_;
    size_t num_of_bits_;
    MemoryAllocatorPtr mem_alloc;
  };
  typedef Bitmap::BitmapPtr BitmapPtr;

  BitmapPtr createBitmap(
      size_t number_of_bits, bool init_value = false, bool initialize = true,
      const hype::ProcessingDeviceMemoryID& mem_id = hype::PD_Memory_0);

  bool operator==(Bitmap& bitmap1, Bitmap& bitmap2);
  bool operator!=(Bitmap& bitmap1, Bitmap& bitmap2);

}  // end namespace CoGaDB
