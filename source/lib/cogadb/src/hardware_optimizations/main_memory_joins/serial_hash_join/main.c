
#include "config.h"

#include "datagen/datagen.h"
#include "join/hashjoin.h"
#include "join/partition.h"
#include "schema.h"
#include "time/time.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <getopt.h>

#include <ctype.h>

struct parameters_t {
  unsigned long r_size;     /**< size of relation R */
  unsigned long s_size;     /**< size of relation S */
  unsigned long s_alphabet; /**< alphabet size used for values in S */
  double s_zipf;            /**< Zipf factor for distribution in S */
  char *part_str;           /**< command line string used to encode
                                 partitioning bit ranges */
  part_bits_t *part_bits;
  bool print_relations; /**< print relations to stdout (mainly for
                             debugging) */
};

static void usage(void) {
  printf("Hash join demo program.\n\n");
  printf("Command line arguments:\n\n");
  printf(
      "  -r SIZE | --size-of-r SIZE      size of (smaller) input relation"
      " R\n");
  printf(
      "  -s SIZE | --size-of-s SIZE      size of (larger) input relation"
      " S\n");
  printf(
      "  -a SIZE | --alphabet-for-s SIZE alphabet size used for values"
      " in S\n");
  printf("                                  (default: size of R)\n");
  printf(
      "  -z VAL | --zipf VAL             Zipf factor for value"
      " distribution in S\n");
  printf("  -p PARTITIONSPEC | --partition PARTITIONSPEC\n");
  printf(
      "                                  partition before computing hash "
      " joins\n");
  printf(
      "                                  (this realizes partitioned hash "
      " join)\n");
  printf("\n");
  printf("Example:\n");
  printf("\n");
  printf("  ./hashjoin -r 1000000 -s 1000000 -p 37-43,44-50\n");
  printf("\n");
  printf(
      "(This will join two relations with 1M tuples each and perform "
      "two-pass\n");
  printf(
      "partitioning using bits 37-43 in the first pass and 44-50 in the "
      "second.)\n");
  printf("\n");
  printf("Have fun!\n");
}

static part_bits_t *parse_part_bits(const char *part_str) {
  const char *s = part_str;
  part_bits_t *ret = NULL;
  part_bits_t *last = NULL;

  /* printf ("Interpreting range `%s'.\n", part_str); */

  while (*s) {
    /* allocate new linked list element */
    if (!ret)
      ret = last = malloc(sizeof(*ret));
    else {
      last->next = malloc(sizeof(*ret));
      last = last->next;
    }

    /* --- populate --- */

    /* read start index */
    last->from_bit = strtol(s, (char **)&s, 10);

    /* consume white space (if any) */
    while (isspace(*s)) s++;

    /* next thing must be a `-' */
    if (*s != '-') {
      fprintf(stderr,
              "Illegal partitioning range; expected `-', "
              "got `%c'\n",
              *s ? *s : ' ');
      exit(EXIT_FAILURE);
    }
    s++; /* skip that `-' */

    /* read end index */
    last->to_bit = strtol(s, (char **)&s, 10);

    /* consume white space (if any) */
    while (isspace(*s)) s++;

    if ((*s != '\0') && (*s != ',')) {
      fprintf(stderr,
              "Illegal partitioning range; expected `-', "
              "got `%c'\n",
              *s);
      exit(EXIT_FAILURE);
    }

    if (*s) s++;

    /* printf ("Found range %u to %u.\n", last->from_bit, last->to_bit); */
  }

  return ret;
}

static struct parameters_t parse_commandline_args(int argc, char **argv) {
  static struct option longopts[] = {
      {"print-relations", no_argument, NULL, 'P'},
      {"alphabet-for-s", required_argument, NULL, 'a'},
      {"help", no_argument, NULL, 'h'},
      {"partition", required_argument, NULL, 'p'},
      {"size-of-r", required_argument, NULL, 'r'},
      {"size-of-s", required_argument, NULL, 's'},
      {"zipf", required_argument, NULL, 'z'},
      {NULL, 0, NULL, 0}};
  int ch;
  struct parameters_t ret = (struct parameters_t){.r_size = 1000000,
                                                  .s_size = 1000000,
                                                  .s_alphabet = 0,
                                                  .s_zipf = 0.,
                                                  .part_str = NULL,
                                                  .print_relations = false};

  while ((ch = getopt_long(argc, argv, "Pa:hp:r:s:z:", longopts, NULL)) != -1) {
    switch (ch) {
      case 'P':
        ret.print_relations = true;
        break;

      case 'a':
        ret.s_alphabet = strtol(optarg, NULL, 10);
        break;

      case 'h':
        usage();
        exit(EXIT_SUCCESS);
        break;

      case 'p':
        ret.part_str = strdup(optarg);
        break;

      case 'r':
        ret.r_size = strtol(optarg, NULL, 10);
        break;

      case 's':
        ret.s_size = strtol(optarg, NULL, 10);
        break;

      case 'z':
        ret.s_size = strtod(optarg, NULL);
        break;

      default:
        usage();
        exit(EXIT_FAILURE);
        break;
    }
  }

  if (ret.s_alphabet == 0) ret.s_alphabet = ret.r_size;

  if (ret.part_str) ret.part_bits = parse_part_bits(ret.part_str);

  return ret;
}

static void print_relation(relation_t rel, char *name) {
  printf("Relation %s:\n\n", name);
  printf("      key    |   value\n");
  printf(" ------------+------------\n");

  for (unsigned long i = 0; i < rel.count; i++)
    printf("  %10u | %10u\n", rel.tuples[i].key, rel.tuples[i].value);

  printf("\n");
}

int main(int argc, char **argv) {
  struct parameters_t params;

  struct timespec t_start;
  struct timespec t_end;

  params = parse_commandline_args(argc, argv);

  my_gettime(&t_start);

  relation_t R = gen_unique(params.r_size);
  relation_t S = gen_zipf(params.s_size, params.s_alphabet, params.s_zipf);

  my_gettime(&t_end);

  printf("Generated data in %lu nsec.\n",
         (t_end.tv_sec * 1000000000L + t_end.tv_nsec) -
             (t_start.tv_sec * 1000000000L + t_start.tv_nsec));

  if (params.print_relations) {
    print_relation(R, "R");
    print_relation(S, "S");
  }

  my_gettime(&t_start);

  unsigned long num_result = hashjoin(R, S, params.part_bits);

  my_gettime(&t_end);

  printf("Full join took %lu nsec.\n",
         (t_end.tv_sec * 1000000000L + t_end.tv_nsec) -
             (t_start.tv_sec * 1000000000L + t_start.tv_nsec));

  printf("%lu join results.\n", num_result);

  return EXIT_SUCCESS;
}
