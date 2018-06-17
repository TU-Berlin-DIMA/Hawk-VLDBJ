/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

namespace hype {
  namespace core {

    class Statistics {
     public:
      static Statistics& Instance() {
        static Statistics statistics;
        return statistics;
      }

      //	void setRecomputationHeuristic(RecomputationHeuristic&
      // recomp_heuristic){
      //		this->recomp_heuristic_=recomp_heuristic;
      //	}

      const bool addObservation(std::string name_of_algorithm,
                                const Measurementpair& mp);  //;
                                                             //	{

      //		//if RecomputationHeuristic.recompute(alg) then
      // algorithm.statistical_method.recomputeApproximationFunction();
      //	}

     private:
      Statistics() {}                         // constructor
      Statistics(const Statistics& stats) {}  // copy constructor

      map<Algorithm&, vector<AlgorithmStatistics> > history_;
      // RecomputationHeuristic& recomp_heuristic_;
      // OptimizationCriteria& opt_criteria_;
    };

  }  // end namespace core
}  // end namespace hype
