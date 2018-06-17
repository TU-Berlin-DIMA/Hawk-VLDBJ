

#include <core/bitmap.hpp>
#include <core/copy_function_factory.hpp>
#include <core/global_definitions.hpp>

namespace CoGaDB {

Bitmap::Bitmap(char* bitmap, size_t num_of_bits,
               const hype::ProcessingDeviceMemoryID& mem_id)
    : bitmap_(bitmap),
      num_of_bits_(num_of_bits),
      mem_alloc(MemoryAllocator<char>::getMemoryAllocator(mem_id)) {
  assert(bitmap_ != NULL);
}
Bitmap::Bitmap(size_t num_of_bits, bool init_value, bool initialize,
               const hype::ProcessingDeviceMemoryID& mem_id)
    : bitmap_(NULL),
      num_of_bits_(num_of_bits),
      mem_alloc(MemoryAllocator<char>::getMemoryAllocator(mem_id)) {
  bitmap_ = mem_alloc->allocate((num_of_bits + 7) / 8);
  if (!bitmap_) throw std::bad_alloc();
  if (initialize) {
    int value = 255 * init_value;
    MemsetFunctionPtr func =
        MemsetFunctionFactory::getMemsetFunction(mem_alloc->getMemoryID());
    assert(func != NULL);
    (*func)(bitmap_, 255, ((num_of_bits + 7) / 8));
  }
  assert(bitmap_ != NULL);
}
Bitmap::~Bitmap() {
  if (bitmap_) {
    mem_alloc->deallocate(bitmap_);
    bitmap_ = NULL;
  }
}
char* Bitmap::data() { return bitmap_; }
size_t Bitmap::size() const { return num_of_bits_; }

hype::ProcessingDeviceMemoryID Bitmap::getMemoryID() const {
  return mem_alloc->getMemoryID();
}

BitmapPtr Bitmap::copy() const { return this->copy(this->getMemoryID()); }
BitmapPtr Bitmap::copy(const hype::ProcessingDeviceMemoryID& mem_id) const {
  BitmapPtr bitmap(new Bitmap(this->num_of_bits_, false, false, mem_id));
  typedef typename CopyFunctionFactory<char>::CopyFunctionPtr CopyFunctionPtr;
  CopyFunctionPtr func = CopyFunctionFactory<char>::getCopyFunction(
      bitmap->getMemoryID(), this->mem_alloc->getMemoryID());
  assert(func != NULL);
  if ((*func)(bitmap->data(), bitmap_, (num_of_bits_ + 7) / 8)) return bitmap;
  return BitmapPtr();
}

bool operator==(Bitmap& bitmap1, Bitmap& bitmap2) {
  if (bitmap1.size() != bitmap2.size()) {
    return false;
  }
  char* bitmap1_data = bitmap1.data();
  char* bitmap2_data = bitmap2.data();

  unsigned int number_of_bits = bitmap1.size();
  for (unsigned int i = 0; i < number_of_bits; ++i) {
    unsigned int current_bit = i & 7;  // i%8;
    unsigned int bitmask = 1 << current_bit;
    char bitmap1_value = bitmap1_data[i / 8];
    char bitmap2_value = bitmap2_data[i / 8];
    if ((bitmap1_value & bitmask) != (bitmap2_value & bitmask)) {
      return false;
    }
  }
  return true;
}
bool operator!=(Bitmap& bitmap1, Bitmap& bitmap2) {
  return !operator==(bitmap1, bitmap2);
}

BitmapPtr createBitmap(size_t number_of_bits, bool init_value, bool initialize,
                       const hype::ProcessingDeviceMemoryID& mem_id) {
  try {
    Bitmap* bitmap;
    bitmap = new Bitmap(number_of_bits, init_value, initialize, mem_id);
    return BitmapPtr(bitmap);
  } catch (std::bad_alloc& e) {
    return BitmapPtr();
  }
}

}  // end namespace CoGaDB
