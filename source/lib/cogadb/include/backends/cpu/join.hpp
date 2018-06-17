/*
 * File:   join.hpp
 * Author: sebastian
 *
 * Created on 19. April 2015, 10:08
 */

#pragma once

#ifndef JOIN_HPP
#define JOIN_HPP

#include <core/base_column.hpp>

namespace CoGaDB {

  template <typename T>
  class CPU_Join {
   public:
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
  };

  template <typename T>
  class CPU_Semi_Join {
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
  };

}  // end namespace CoGaDB

#endif /* JOIN_HPP */
