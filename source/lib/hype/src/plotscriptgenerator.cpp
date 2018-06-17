
#include <config/configuration.hpp>
#include <config/global_definitions.hpp>
#include <core/plotscriptgenerator.hpp>

#include <cmath>
#include <fstream>
#include <iostream>

#include <boost/filesystem.hpp>

namespace hype {
namespace core {

using namespace std;

bool PlotScriptGenerator::create(
    const std::string& title, const std::string& xlabel,
    const std::string& ylabel, const std::string& operation_name,
    const std::vector<std::string>& algorithm_names) {
  const std::string output_dir_name = "output";
  if (!boost::filesystem::exists(output_dir_name)) {
    if (!quiet) cout << "create Directory '" << output_dir_name << "'" << endl;
    if (!boost::filesystem::create_directory(output_dir_name)) {
      cout << "HyPE Library: Failed to created Output Directory '"
           << output_dir_name << "' for operation statistics, skipping the "
                                 "write operation for statistical data for "
                                 "operation '"
           << operation_name << "'" << endl;
    }
  }
  std::string dir_name = output_dir_name + "/";

  dir_name += operation_name;
  dir_name += "/";

  if (!boost::filesystem::create_directory(dir_name) && !quiet)
    std::cout << "Directory '" << dir_name << "' already exists!" << std::endl;

  //		dir_name+=algorithm_name;
  //
  //		if(!boost::filesystem::create_directory(dir_name));
  //			std::cout << "Directory '" << dir_name << "' already
  // exists!"
  //<<
  // std::endl;

  //		std::string basic_file_name=dir_name+"/";
  //		basic_file_name+=algorithm_name;

  std::string file_name_measurement_times = dir_name + "execution_times.plt";
  if (!quiet)
    std::cout << "Creating gnuplot script file '"
              << file_name_measurement_times.c_str() << "'" << std::endl;
  std::fstream file(file_name_measurement_times.c_str(),
                    std::ios_base::out | std::ios_base::trunc);

  file << "set title '" << title << "'" << std::endl
       << "set xtics nomirror" << std::endl
       << "set ytics nomirror" << std::endl
       << "set xlabel '" << xlabel << "'" << std::endl
       << "set ylabel '" << ylabel << "'" << std::endl
       << "set key top left Left reverse samplen 1" << std::endl
       << std::endl
       << "plot ";

  for (unsigned int i = 0; i < algorithm_names.size(); i++) {
    file << "'" << algorithm_names[i] << "/"
         << "measurement_pairs.data' using 1:2 title \"" << algorithm_names[i]
         << " measured execution times\" with points, \\" << std::endl
         << "'" << algorithm_names[i] << "/"
         << "measurement_pairs.data' using 1:3 title \"" << algorithm_names[i]
         << " estimated execution times\" with points";
    if (i < algorithm_names.size() - 1) file << ", \\" << std::endl;
  }
  file << std::endl
       << std::endl
       << "set output \"execution_times.pdf\"" << std::endl
       << "set terminal pdfcairo font \"Helvetica,9\"" << std::endl
       << "replot" << std::endl;
  //
  //

  return true;
}

bool PlotScriptGenerator::createRelativeErrorScript(
    const std::string& operation_name,
    const std::vector<std::string>& algorithm_names) {
  std::string dir_name = "output/";
  dir_name += operation_name;
  dir_name += "/";

  if (!boost::filesystem::create_directory(dir_name) && !quiet)
    std::cout << "Directory '" << dir_name << "' already exists!" << std::endl;

  //		dir_name+=algorithm_name;
  //
  //		if(!boost::filesystem::create_directory(dir_name));
  //			std::cout << "Directory '" << dir_name << "' already
  // exists!"
  //<<
  // std::endl;

  //		std::string basic_file_name=dir_name+"/";
  //		basic_file_name+=algorithm_name;

  std::string file_name_measurement_times = dir_name + "relative_errors.plt";
  if (!quiet)
    std::cout << "Creating gnuplot script file '"
              << file_name_measurement_times.c_str() << "'" << std::endl;
  std::fstream file(file_name_measurement_times.c_str(),
                    std::ios_base::out | std::ios_base::trunc);

  file << "set title 'relative estimation errors for operation "
       << operation_name << "'" << std::endl
       << "set xtics nomirror" << std::endl
       << "set ytics nomirror" << std::endl
       << "set xlabel 'iterations'" << std::endl
       << "set ylabel 'absolute relative estmation error'" << std::endl
       << "set logscale y" << std::endl
       //<< "set key top left Left reverse samplen 1" << std::endl
       << "set key below" << std::endl
       << "set key box" << std::endl
       << std::endl
       << "plot ";

  for (unsigned int i = 0; i < algorithm_names.size(); i++) {
    file << "'" << algorithm_names[i] << "/"
         << "relative_errors.data' using 1:(abs($2)) title \""
         << algorithm_names[i] << " relative error\" with points";
    if (i < algorithm_names.size() - 1) {
      file << ", \\" << std::endl;
    }
  }
  file << std::endl
       << std::endl
       << "set output \"relative_errors.pdf\"" << std::endl
       << "set terminal pdfcairo font \"Helvetica,9\"" << std::endl
       << "replot" << std::endl;
  //
  //

  return true;
}

bool PlotScriptGenerator::createAverageRelativeErrorScript(
    const std::string& operation_name,
    const std::vector<std::string>& algorithm_names) {
  std::string dir_name = "output/";
  dir_name += operation_name;
  dir_name += "/";

  if (!boost::filesystem::create_directory(dir_name) && !quiet)
    std::cout << "Directory '" << dir_name << "' already exists!" << std::endl;

  //		dir_name+=algorithm_name;
  //
  //		if(!boost::filesystem::create_directory(dir_name));
  //			std::cout << "Directory '" << dir_name << "' already
  // exists!"
  //<<
  // std::endl;

  //		std::string basic_file_name=dir_name+"/";
  //		basic_file_name+=algorithm_name;

  std::string file_name_measurement_times =
      dir_name + "average_relative_errors.plt";
  if (!quiet)
    std::cout << "Creating gnuplot script file '"
              << file_name_measurement_times.c_str() << "'" << std::endl;
  std::fstream file(file_name_measurement_times.c_str(),
                    std::ios_base::out | std::ios_base::trunc);

  file << "set title 'average relative estimation errors for operation "
       << operation_name << "'" << std::endl
       << "set xtics nomirror" << std::endl
       << "set ytics nomirror" << std::endl
       << "set xlabel 'iterations'" << std::endl
       << "set ylabel 'average absolute relative estmation error'" << std::endl
       << "set logscale y" << std::endl
       //<< "set key top left Left reverse samplen 1" << std::endl
       << "set key below" << std::endl
       << "set key box" << std::endl
       << std::endl
       << "plot ";

  for (unsigned int i = 0; i < algorithm_names.size(); i++) {
    file << "'" << algorithm_names[i] << "/"
         << "average_relative_errors.data' using 1:(abs($2)) title \""
         << algorithm_names[i] << " relative error\" with points";
    if (i < algorithm_names.size() - 1) {
      file << ", \\" << std::endl;
    }
  }
  file << std::endl
       << std::endl
       << "set output \"average_relative_errors.pdf\"" << std::endl
       << "set terminal pdfcairo font \"Helvetica,9\"" << std::endl
       << "replot" << std::endl;
  //
  //

  return true;
}

bool PlotScriptGenerator::createWindowedAverageRelativeErrorScript(
    const std::string& operation_name,
    const std::vector<std::string>& algorithm_names) {
  std::string dir_name = "output/";
  dir_name += operation_name;
  dir_name += "/";

  if (!boost::filesystem::create_directory(dir_name) && !quiet)
    std::cout << "Directory '" << dir_name << "' already exists!" << std::endl;

  //		dir_name+=algorithm_name;
  //
  //		if(!boost::filesystem::create_directory(dir_name));
  //			std::cout << "Directory '" << dir_name << "' already
  // exists!"
  //<<
  // std::endl;

  //		std::string basic_file_name=dir_name+"/";
  //		basic_file_name+=algorithm_name;

  std::string file_name_measurement_times =
      dir_name + "windowed_average_relative_errors.plt";
  if (!quiet)
    std::cout << "Creating gnuplot script file '"
              << file_name_measurement_times.c_str() << "'" << std::endl;
  std::fstream file(file_name_measurement_times.c_str(),
                    std::ios_base::out | std::ios_base::trunc);

  file
      << "set title 'windowed average relative estimation errors for operation "
      << operation_name << " (Window size="
      << Runtime_Configuration::instance().getRelativeErrorWindowSize() << ")'"
      << std::endl
      << "set xtics nomirror" << std::endl
      << "set ytics nomirror" << std::endl
      << "set xlabel 'iterations'" << std::endl
      << "set ylabel 'windowed average absolute relative estmation error'"
      << std::endl
      << "set logscale y" << std::endl
      //<< "set key top left Left reverse samplen 1" << std::endl
      << "set key below" << std::endl
      << "set key box" << std::endl
      << std::endl
      << "plot ";

  for (unsigned int i = 0; i < algorithm_names.size(); i++) {
    file << "'" << algorithm_names[i] << "/"
         << "windowed_average_relative_errors.data' using 1:(abs($2)) title \""
         << algorithm_names[i] << " relative error\" with points";
    if (i < algorithm_names.size() - 1) file << ", \\" << std::endl;
  }
  file << std::endl
       << std::endl
       << "set output \"windowed_average_relative_errors.pdf\"" << std::endl
       << "set terminal pdfcairo font \"Helvetica,9\"" << std::endl
       << "replot" << std::endl;
  //
  //

  return true;
}

bool PlotScriptGenerator::create_3d_plot(
    const std::string& title, const std::string& xlabel,
    const std::string& ylabel, const std::string& directory_path,
    const std::vector<std::string>& algorithm_names) {
  std::string dir_name = directory_path + "/";

  if (!boost::filesystem::create_directory(dir_name) && !quiet)
    std::cout << "Directory '" << dir_name << "' already exists!" << std::endl;

  //		dir_name+=algorithm_name;
  //
  //		if(!boost::filesystem::create_directory(dir_name));
  //			std::cout << "Directory '" << dir_name << "' already
  // exists!"
  //<<
  // std::endl;

  //		std::string basic_file_name=dir_name+"/";
  //		basic_file_name+=algorithm_name;

  std::string file_name_measurement_times = dir_name + "execution_times.plt";
  if (!quiet)
    std::cout << "Creating gnuplot script file '"
              << file_name_measurement_times.c_str() << "'" << std::endl;
  std::fstream file(file_name_measurement_times.c_str(),
                    std::ios_base::out | std::ios_base::trunc);

  file << "set title '" << title << "'" << std::endl
       << "set xtics nomirror" << std::endl
       << "set ytics nomirror" << std::endl
       << "set xlabel '" << xlabel << "'" << std::endl
       << "set ylabel '" << ylabel << "'" << std::endl
       << "set zlabel 'execution time in ns'" << std::endl
       << "set key below" << std::endl
       << "set key box" << std::endl
       //<< "set key top left Left reverse samplen 1" << std::endl
       << "set hidden3d" << std::endl
       << "set view 75,40,1.0,2.5" << std::endl
       << std::endl
       << "splot ";

  for (unsigned int i = 0; i < algorithm_names.size(); i++) {
    file << "'" << algorithm_names[i] << ".data' using 1:2:3 title \""
         << algorithm_names[i] << " measured execution times\" with points";
    //			file <<"'"<< algorithm_names[i] << ".data' using 1:2:3
    // title
    //\""
    //<<
    // algorithm_names[i] << " measured execution times\" with points,
    //\\"<<std::endl
    //				  <<"'"<< algorithm_names[i] << ".data' using
    // 1:2:4
    // title
    //\""
    //<<
    // algorithm_names[i] << " estimated execution times\" with points";
    if (i < algorithm_names.size() - 1) file << ", \\" << std::endl;
  }
  file << ", \\" << std::endl
       << "'complete_execution_history.data' using 1:2:3 title \"decision\" "
          "with points, \\"
       << std::endl
       << "'complete_execution_history.data' using 1:2:4 title \"estimation\" "
          "with points"
       << std::endl;

  //	   file << std::endl << std::endl
  //			  << "set output \"execution_times.pdf\"" << std::endl
  //			  << "set terminal pdfcairo font \"Helvetica,7\"" <<
  // std::endl
  //			  << "replot" << std::endl;
  //
  //

  return true;
}

}  // end namespace core
}  // end namespace hype
