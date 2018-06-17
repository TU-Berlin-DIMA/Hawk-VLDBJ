/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   ocl_api.hpp
 * Author: sebastian
 *
 * Created on 26. Februar 2016, 20:35
 */

#ifndef OCL_API_HPP
#define OCL_API_HPP

#include <query_compilation/ocl_api.h>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/program.hpp>
#include <boost/shared_ptr.hpp>

#include <atomic>

struct OCL_Execution_Context {
  OCL_Execution_Context(
      const boost::compute::context& context,
      const std::vector<boost::compute::command_queue>& compute_queues,
      const std::vector<boost::compute::command_queue>&
          copy_host_to_device_queues,
      const std::vector<boost::compute::command_queue>&
          copy_device_to_host_queues,
      const boost::compute::program& program);

  ~OCL_Execution_Context();
  cl_context context;

  std::vector<boost::compute::command_queue> compute_queues;
  std::vector<boost::compute::command_queue> copy_host_to_device_queues;
  std::vector<boost::compute::command_queue> copy_device_to_host_queues;
  std::atomic_uint_fast32_t thread_access_index_counter;
  boost::compute::program program;
};

typedef boost::shared_ptr<OCL_Execution_Context> OCL_Execution_ContextPtr;

void ocl_api_reset_thread_local_variables();

#endif /* OCL_API_HPP */
