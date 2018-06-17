/***********************************************************************************************************
Copyright (c) 2012, Sebastian Breß, Otto-von-Guericke University of Magdeburg,
Germany. All rights reserved.

This program and accompanying materials are made available under the terms of
the
GNU LESSER GENERAL PUBLIC LICENSE - Version 3,
http://www.gnu.org/licenses/lgpl-3.0.txt
***********************************************************************************************************/
#pragma once

#include <string>
#include <vector>

#include <config/global_definitions.hpp>

namespace hype {
  namespace core {

    class PlotScriptGenerator {
     public:
      static bool create(const std::string& title, const std::string& xlabel,
                         const std::string& ylabel,
                         const std::string& operation_name,
                         const std::vector<std::string>& algorithm_names);

      static bool createRelativeErrorScript(
          const std::string& operation_name,
          const std::vector<std::string>& algorithm_names);

      static bool createAverageRelativeErrorScript(
          const std::string& operation_name,
          const std::vector<std::string>& algorithm_names);

      static bool createWindowedAverageRelativeErrorScript(
          const std::string& operation_name,
          const std::vector<std::string>& algorithm_names);

      static bool create_3d_plot(
          const std::string& title, const std::string& xlabel,
          const std::string& ylabel, const std::string& directory_path,
          const std::vector<std::string>& algorithm_names);
    };

  }  // end namespace core
}  // end namespace hype
