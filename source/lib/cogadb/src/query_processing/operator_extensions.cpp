
#include <core/base_column.hpp>
#include <query_processing/operator_extensions.hpp>

namespace CoGaDB {
namespace query_processing {

bool BitmapOperator::hasResultBitmap() { return (cpu_bitmap_ != NULL); }

BitmapPtr BitmapOperator::getResultBitmap() { return cpu_bitmap_; }

void BitmapOperator::releaseResultData() { this->cpu_bitmap_.reset(); }

bool PositionListOperator::hasResultPositionList() { return (tids_ != NULL); }

PositionListPtr PositionListOperator::getResultPositionList() { return tids_; }

void PositionListOperator::releaseResultData() { this->tids_.reset(); }

}  // end namespace query_processing
}  // end namespace CoGaDB
