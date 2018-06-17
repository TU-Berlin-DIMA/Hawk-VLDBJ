/*
 * File:   fetch_join.hpp
 * Author: sebastian
 *
 * Created on 3. Januar 2015, 09:05
 */

#ifndef FETCH_JOIN_HPP
#define FETCH_JOIN_HPP

#include <core/base_column.hpp>

namespace CoGaDB {

  class GPU_FetchJoin {
   public:
    static const PositionListPtr tid_fetch_join(JoinIndexPtr join_index,
                                                PositionListPtr pk_table_tids,
                                                const FetchJoinParam&);

    static const PositionListPairPtr fetch_join(JoinIndexPtr join_index,
                                                PositionListPtr pk_table_tids,
                                                const FetchJoinParam&);

    static const BitmapPtr bitmap_fetch_join(JoinIndexPtr join_index,
                                             PositionListPtr pk_table_tids,
                                             const FetchJoinParam&);
  };

}  // end namespace CoGaDB

#endif /* FETCH_JOIN_HPP */
