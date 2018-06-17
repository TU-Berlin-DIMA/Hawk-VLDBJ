/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once
#include <vector>

namespace hype {
  namespace util {

    template <class T, class TAl>
    inline T* begin_ptr(std::vector<T, TAl>& v) {
      return v.empty() ? NULL : &v[0];
    }

    template <class T, class TAl>
    inline const T* begin_ptr(const std::vector<T, TAl>& v) {
      return v.empty() ? NULL : &v[0];
    }

  }  // end namespace util
}  // end namespace hype
