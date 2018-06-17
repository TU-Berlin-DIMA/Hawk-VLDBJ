/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <boost/shared_ptr.hpp>
#include <iostream>
#include <string>
#include <vector>

// hype includes
#include <config/global_definitions.hpp>
#include <core/factory.hpp>
#include <core/time_measurement.hpp>

namespace hype {
  namespace core {

    class Algorithm;  // forward declaration

    class RecomputationHeuristic_Internal {
     public:
      bool recompute(Algorithm& algorithm);
      virtual bool internal_recompute(Algorithm& algorithm) = 0;
      virtual ~RecomputationHeuristic_Internal() {}

      std::string getName() const throw() { return this->name_; }

     protected:
      RecomputationHeuristic_Internal(const std::string& name);

     private:
      std::string name_;
      unsigned int length_of_training_phase_;
      bool is_initial_approximation_function_computed_;
      unsigned int samplecounter_;
    };

    // factory function
    const boost::shared_ptr<RecomputationHeuristic_Internal>
    getNewRecomputationHeuristicbyName(std::string name);

    typedef core::Factory<RecomputationHeuristic_Internal, std::string>
        RecomputationHeuristicFactory;  // aFactory;
    // typedef Loki::Singleton<RecomputationHeuristicFactory>
    // RecomputationHeuristicFactorySingleton;

    class RecomputationHeuristicFactorySingleton {
     public:
      static RecomputationHeuristicFactory& Instance();
    };

  }  // end namespace core
}  // end namespace hype
