#pragma once
#include <string>
#include <vector>

namespace CoGaDB {

  const std::vector<std::string> getFilesinDirectory(std::string dname);

  bool is_regular_file(const std::string& path);

  const std::string getPathToHomeConfigDir();

  bool createPIDFile();

  bool deletePIDFile();

}  // end namespace CogaDB
