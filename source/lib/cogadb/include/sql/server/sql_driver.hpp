#ifndef SQL_DRIVER_HPP
#define SQL_DRIVER_HPP

#include <parser/client.hpp>
#include "sql/server/sql_parsetree.hpp"
#include "sql_parser.hpp"

#include <exception>
#include <iostream>
#include <string>

namespace CoGaDB {
  namespace SQL {

    namespace Scanner {
      typedef void *buffer;

      Parser::token_type lex(Parser::semantic_type *yylval_param,
                             type yyscanner);
    }

    class Driver {
      Scanner::buffer buffer;
      /* FIXME: should not be public */
     public:
      std::istream *istream;

     private:
      Scanner::type scanner;

      Parser parser;

      ParseTree::SequencePtr result;

     public:
      class ParseError : public std::exception {
       public:
        ParseError() throw() : std::exception(), message() {}
        ParseError(const std::string &mes) throw()
            : std::exception(), message(mes) {}
        const char *what() const throw() {
          if (message.empty()) {
            return "Error during parsing";
          } else {
            std::string error = "Error: ";
            error += message;
            return error.c_str();
          }
        }
        virtual ~ParseError() throw() {}

       private:
        std::string message;
      };

      Driver();
      ~Driver();

      ParseTree::SequencePtr parse(std::istream &is);
      ParseTree::SequencePtr parse(const std::string &src);

     private:
      /*
       * defined in sql-scanner.lpp
       */
      void init_scan();

      void set_input(std::istream &is);
      void set_input(const std::string &src);

      void destroy_scan();

      void error(const std::string &m);

      inline void unsupported(void) { error("Unsupported language construct"); }

      /*
       * Friends
       */
      friend Parser::token_type Scanner::lex(Parser::semantic_type *,
                                             Scanner::type);
      friend class Parser;
    };

    bool commandlineExec(const std::string &input, ClientPtr client);
    bool commandlineExplain(const std::string &input, ClientPtr client);
    // bool commandlineExplainStatements(const std::string &input);
    bool commandlineExplainStatementsWithoutOptimization(
        const std::string &input, ClientPtr client);
    bool commandlineExplainStatementsWithOptimization(const std::string &input,
                                                      ClientPtr client);

    TablePtr executeSQL(const std::string &input, ClientPtr client);

  } /* namespace SQL */
} /* namespace CoGaDB */

#endif
