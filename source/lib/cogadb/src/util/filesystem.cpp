

#include <pwd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <errno.h>
#include <sys/stat.h>

#include <dirent.h>
#include <unistd.h>

#include <err.h>
#include <errno.h>
#include <string.h>

#include <fstream>
#include <iostream>
#include <list>
#include <string>
#include <vector>

#include <core/variable_manager.hpp>
#include <util/filesystem.hpp>

#include <boost/lexical_cast.hpp>

namespace CoGaDB {

enum {
  WALK_OK = 0,
  WALK_BADPATTERN,
  WALK_NAMETOOLONG,
  WALK_BADIO,
};

#define WS_NONE 0
#define WS_RECURSIVE (1 << 0)
#define WS_DEFAULT WS_RECURSIVE
#define WS_FOLLOWLINK (1 << 1) /* follow symlinks */
#define WS_DOTFILES (1 << 2)   /* per unix convention, .file is hidden */
#define WS_MATCHDIRS (1 << 3)  /* if pattern is used on dir names too */

const std::vector<std::string> getFilesinDirectory(std::string dname) {
  std::vector<std::string> found_files;
  dname += "/";
  struct dirent* dent;
  DIR* dir;
  struct stat st;
  char fn[FILENAME_MAX];
  int res = WALK_OK;
  int len = strlen(dname.c_str());
  if (len >= FILENAME_MAX - 1) return found_files;

  strcpy(fn, dname.c_str());
  fn[len++] = '/';

  if (!(dir = opendir(dname.c_str()))) {
    std::cout << "can't open " << dname << std::endl;
    return found_files;
  }

  errno = 0;
  while ((dent = readdir(dir))) {
    // if (!(spec & WS_DOTFILES) && dent->d_name[0] == '.')
    //	continue;
    if (!strcmp(dent->d_name, ".") || !strcmp(dent->d_name, "..")) continue;

    strncpy(fn + len, dent->d_name, FILENAME_MAX - len);
    if (lstat(fn, &st) == -1) {
      std::cout << "can't stat " << fn << std::endl;
      res = WALK_BADIO;
      continue;
    }

    /* don't follow symlink unless told so */
    if (S_ISLNK(st.st_mode))  //&& !(spec & WS_FOLLOWLINK))
      continue;

    /* will be false for symlinked dirs */
    if (S_ISDIR(st.st_mode)) {
      /* recursively follow dirs */
      // if ((spec & WS_RECURSIVE))
      //	walk_recur(fn, reg, spec);

      // if (!(spec & WS_MATCHDIRS)) continue;
    }

    found_files.push_back(dent->d_name);
  }

  if (dir) closedir(dir);
  if (res == WALK_BADIO)
    std::cout << "IO Error Occured while traversing directory '" << dname << "'"
              << std::endl;
  return found_files;
}

bool is_regular_file(const std::string& path) {
  using namespace std;
  int status;
  struct stat st_buf;

  // Get the status of the file system object.
  status = stat(path.c_str(), &st_buf);
  if (status != 0) {
    cout << "Error, errno = " << errno << endl;
    return false;
  }

  // Tell us what it is then exit.
  return bool(S_ISREG(st_buf.st_mode));
  /*
  {
          printf ("%s is a regular file.\n", argv[1]);
      }
      if (S_ISDIR (st_buf.st_mode)) {
          printf ("%s is a directory.\n", argv[1]);
      }

      return 0;*/
}

const std::string getPathToHomeConfigDir() {
  passwd* pw = getpwuid(getuid());
  std::string path(pw->pw_dir);
  path += "/.cogadb/";
  return path;
}

bool createPIDFile() {
  if (VariableManager::instance().getVariableValueString(
          "path_to_database_farm") == "" ||
      VariableManager::instance().getVariableValueString(
          "name_of_current_databse") == "") {
    return false;
  }

  std::string pidfilePath = VariableManager::instance().getVariableValueString(
      "path_to_database_farm");
  pidfilePath.append(VariableManager::instance().getVariableValueString(
      "name_of_current_databse"));
  pidfilePath.append("/pid");
  std::cout << "Write PID to " << pidfilePath << std::endl;
  std::remove(pidfilePath.c_str());
  std::ofstream pidFile(pidfilePath.c_str());
  if (pidFile.is_open()) {
    pidFile << ::getpid();
    return true;
  } else {
    std::cout << "Could not write a pid file" << std::endl;
    return false;
  }
}

bool deletePIDFile() {
  if (VariableManager::instance().getVariableValueString(
          "path_to_database_farm") == "" ||
      VariableManager::instance().getVariableValueString(
          "name_of_current_databse") == "") {
    return false;
  }

  std::string pidfilePath = VariableManager::instance().getVariableValueString(
      "path_to_database_farm");
  pidfilePath.append(VariableManager::instance().getVariableValueString(
      "name_of_current_databse"));
  pidfilePath.append("/pid");
  return std::remove(pidfilePath.c_str());
}

}  // end namespace CogaDB
