#pragma once

#include <core/base_column.hpp>
#include <core/bitmap.hpp>

namespace CoGaDB {
  namespace query_processing {

    class BitmapOperator {
     public:
      bool hasResultBitmap();
      BitmapPtr getResultBitmap();
      void releaseResultData();

     protected:
      BitmapPtr cpu_bitmap_;
    };

    class PositionListOperator {
     public:
      bool hasResultPositionList();
      PositionListPtr getResultPositionList();
      void releaseResultData();

     protected:
      //            PositionListPtr cpu_tids_;
      PositionListPtr tids_;
    };

  }  // end namespace CoGaDB
}  // end namespace CoGaDB
