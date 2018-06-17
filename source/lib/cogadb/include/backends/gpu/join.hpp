/*
 * File:   join.hpp
 * Author: sebastian
 *
 * Created on 7. Januar 2015, 15:44
 */

#pragma once

#ifndef GPU_JOIN_HPP
#define GPU_JOIN_HPP

//#include "core/base_table.hpp"

namespace CoGaDB {

  template <typename T>
  class GPU_Join {
   public:
    //        static const PositionListPairPtr join(T* join_column1,
    //                size_t left_num_elements,
    //                T* join_column2,
    //                size_t right_num_elements,
    //                const JoinParam&);

    typedef const PositionListPairPtr (*JoinFunctionPtr)(
        T* join_column1, size_t left_num_elements, T* join_column2,
        size_t right_num_elements, const JoinParam& param);

    static JoinFunctionPtr get(const JoinType& join_type);

   private:
    static const PositionListPairPtr inner_join(T* join_column1,
                                                size_t left_num_elements,
                                                T* join_column2,
                                                size_t right_num_elements,
                                                const JoinParam&);

    static const PositionListPairPtr left_outer_join(T* join_column1,
                                                     size_t left_num_elements,
                                                     T* join_column2,
                                                     size_t right_num_elements,
                                                     const JoinParam&);

    static const PositionListPairPtr right_outer_join(T* join_column1,
                                                      size_t left_num_elements,
                                                      T* join_column2,
                                                      size_t right_num_elements,
                                                      const JoinParam&);

    static const PositionListPairPtr full_outer_join(T* join_column1,
                                                     size_t left_num_elements,
                                                     T* join_column2,
                                                     size_t right_num_elements,
                                                     const JoinParam&);

   private:
    static const PositionListPairPtr binary_search_join(
        T* join_column1, T* join_column2, size_t left_num_elements,
        size_t right_num_elements, const JoinParam&);
  };

  template <typename T>
  class GPU_Semi_Join {
   public:
    typedef const PositionListPtr (*TIDSemiJoinFunctionPtr)(
        T* join_column1, size_t left_num_elements, T* join_column2,
        size_t right_num_elements, const JoinParam& param);

    typedef const BitmapPtr (*BitmapSemiJoinFunctionPtr)(
        T* join_column1, size_t left_num_elements, T* join_column2,
        size_t right_num_elements, const JoinParam& param);

    static TIDSemiJoinFunctionPtr getTIDSemiJoin(const JoinType& join_type);

    static BitmapSemiJoinFunctionPtr getBitmapSemiJoin(
        const JoinType& join_type);

   private:
    static const PositionListPtr tid_left_semi_join(T* join_column1,
                                                    size_t left_num_elements,
                                                    T* join_column2,
                                                    size_t right_num_elements,
                                                    const JoinParam&);

    static const PositionListPtr tid_right_semi_join(T* join_column1,
                                                     size_t left_num_elements,
                                                     T* join_column2,
                                                     size_t right_num_elements,
                                                     const JoinParam&);

    static const PositionListPtr tid_left_anti_semi_join(
        T* join_column1, size_t left_num_elements, T* join_column2,
        size_t right_num_elements, const JoinParam&);

    static const PositionListPtr tid_right_anti_semi_join(
        T* join_column1, size_t left_num_elements, T* join_column2,
        size_t right_num_elements, const JoinParam&);

    static const BitmapPtr bitmap_left_semi_join(T* join_column1,
                                                 size_t left_num_elements,
                                                 T* join_column2,
                                                 size_t right_num_elements,
                                                 const JoinParam&);

    static const BitmapPtr bitmap_right_semi_join(T* join_column1,
                                                  size_t left_num_elements,
                                                  T* join_column2,
                                                  size_t right_num_elements,
                                                  const JoinParam&);

    static const BitmapPtr bitmap_left_anti_semi_join(T* join_column1,
                                                      size_t left_num_elements,
                                                      T* join_column2,
                                                      size_t right_num_elements,
                                                      const JoinParam&);

    static const BitmapPtr bitmap_right_anti_semi_join(
        T* join_column1, size_t left_num_elements, T* join_column2,
        size_t right_num_elements, const JoinParam&);

    static const PositionListPtr d_SemiJoin(T* d_build, size_t numBuild,
                                            T* d_probe, size_t numProbe,
                                            bool anti, const JoinParam& param);

    static const BitmapPtr d_BitmapSemiJoin(T* d_build, size_t numBuild,
                                            T* d_probe, size_t numProbe,
                                            bool anti, const JoinParam& param);
  };

}  // end namespace CoGaDB

#endif /* GPU_JOIN_HPP */
