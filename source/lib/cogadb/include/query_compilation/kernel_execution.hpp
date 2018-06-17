#include <list>
#include <string>
#include <vector>

class KernelExecution {
 private:
  std::string name;
  std::string function_header;
  std::list<std::string> kernel_code_upper;
  std::list<std::string> kernel_code_lower;

  std::vector<std::string> parameter_types;
  std::vector<std::string> parameter_names;
  std::vector<std::string> parameter_call_expressions;

  int block_size;
  std::string index_variable_type;
  std::string tuple_id_var_name;
  size_t limit;
  bool use_selected;
  bool codeIsProduced;

  void produceFrameCode();

 public:
  KernelExecution();

  void init(std::string name, int blockSize, std::string index_variable_type,
            std::string tuple_id_var_name, size_t limit, bool use_selected);

  void addParameter(std::string type, std::string variable_name);

  void addParameter(std::string type, std::string variable_name,
                    std::string call_expression);

  void prepend_upper(std::string);
  void append_upper(std::string);
  void append_lower(std::string);

  std::string getKernelCode();
  std::string getInvocationCode();
  int getBlockSize();
};
