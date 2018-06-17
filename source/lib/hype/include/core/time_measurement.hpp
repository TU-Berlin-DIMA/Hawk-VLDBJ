/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <config/exports.hpp>

#include <stdint.h>
#include <vector>

#include <boost/serialization/access.hpp>

namespace hype {
  namespace core {

    class HYPE_EXPORT EstimatedTime {
     public:
      EstimatedTime();
      explicit EstimatedTime(double time_in_nanoseconds);

      double getTimeinNanoseconds() const;
      friend class boost::serialization::access;
      // When the class Archive corresponds to an output archive, the
      // & operator is defined similar to <<.  Likewise, when the class Archive
      // is a type of input archive the & operator is defined similar to >>.
      template <class Archive>
      void serialize(Archive& ar, const unsigned int version) {
        ar& value_in_nanoseconds_;
      }

     private:
      double value_in_nanoseconds_;
    };

    class HYPE_EXPORT MeasuredTime {
     public:
      MeasuredTime();
      explicit MeasuredTime(double time_in_nanoseconds);

      double getTimeinNanoseconds() const;
      friend class boost::serialization::access;
      // When the class Archive corresponds to an output archive, the
      // & operator is defined similar to <<.  Likewise, when the class Archive
      // is a type of input archive the & operator is defined similar to >>.
      template <class Archive>
      void serialize(Archive& ar, const unsigned int version) {
        ar& value_in_nanoseconds_;
      }

     private:
      double value_in_nanoseconds_;
    };

    typedef std::vector<double> Tuple;
    typedef size_t (*MemoryCostModelFuncPtr)(const Tuple&);

    HYPE_EXPORT uint64_t getTimestamp();

    // uint64_t getTimestamp()
    //{
    //    timespec ts;
    //    //clock_gettime(CLOCK_REALTIME, &ts);
    //    //return (uint64_t)ts.tv_sec * 1000000LL + (uint64_t)ts.tv_nsec /
    //    1000LL;
    //    clock_gettime(CLOCK_REALTIME, &ts);
    // 	 return (uint64_t)ts.tv_sec * 1000000000LL + (uint64_t)ts.tv_nsec;
    //}

  }  // end namespace core
}  // end namespace hype
