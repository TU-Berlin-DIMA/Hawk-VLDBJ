/*
 * File:   bitmap_set_operation.hpp
 * Author: sebastian
 *
 * Created on 3. Januar 2015, 10:41
 */

#ifndef BITMAP_SET_OPERATION_HPP
#define BITMAP_SET_OPERATION_HPP

//#include <core/global_definitions.hpp>
//#include <core/bitmap.hpp>
#include <core/base_column.hpp>

namespace CoGaDB {

  class GPU_BitmapSetOperation {
   public:
    static const BitmapPtr computeBitmapSetOperation(
        BitmapPtr left_bitmap, BitmapPtr right_bitmap,
        const BitmapOperationParam& param);

    //         const BitmapPtr computeBitmapAND(BitmapPtr left_bitmap,
    //                                            BitmapPtr right_bitmap,
    //                                            const ProcessorSpecification&
    //                                            proc_spec);
    //
    //         const BitmapPtr computeBitmapOR(BitmapPtr left_bitmap,
    //                                           BitmapPtr right_bitmap,
    //                                           const ProcessorSpecification&
    //                                           proc_spec);
  };

}  // end namespace CoGaDB

#endif /* BITMAP_SET_OPERATION_HPP */
