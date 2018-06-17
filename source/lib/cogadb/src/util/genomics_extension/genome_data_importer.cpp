//
// Created by basti on 06.07.15.
//

#include <core/global_definitions.hpp>

#ifdef BAM_FOUND

#include <util/genomics_extension/genome_data_importer.hpp>

namespace CoGaDB {

using namespace std;

GenomeDataImporter::GenomeDataImporter(ostream &_out, bool _compress_data,
                                       bool _verbose)
    : out(_out), verbose(_verbose), compress_data(_compress_data) {}

GenomeDataImporter::~GenomeDataImporter() {}

/*
* Input:
* uint32_t *cigar ... CIGAR string to convert
* int cigarLength ... length of CIGAR string
* int startPos ... start position of alignment
*
* Output:
* uint32_t* positions ... corrected alignment positions for every base in a read
*according to the CIGAR string
* uint32_t* insert_offsets ... insert offsets for every base ion a read
*according
*to the CIGAR string
* uint32_t* deleted_positions ... positions where deleted bases 'X' have to be
*inserted
* int* start_offset ... offset , int* end_offset
*/
/* array to store positions */
/* array to store insert_offsets */
/* array to store positions for deletions */

/* soft clip offsets */

void GenomeDataImporter::_convertCIGAR(uint32_t *cigar, uint32_t cigarLength,
                                       int startPos,
                                       cigar_conversion *converted_cigar) {
  std::ignore = startPos;

  bool hard_clip_not_present = 1;
  uint32_t cur_index = 0;
  uint32_t cur_del_index = 0;
  uint32_t i, j;
  uint32_t pos = 0;
  for (i = 0; i < cigarLength; ++i) {
    switch (bam_cigar_op(*cigar)) {
      case BAM_CMATCH: /* M = match or mismatch*/
      case BAM_CEQUAL: /* equals = match */
      case BAM_CDIFF:  /* X = mismatch */
                       /* base matches a reference base (does not mean that
                    they are equal) -> increment positions, offset = 0 */
        for (j = 0; j < bam_cigar_oplen(*cigar); j++) {
          converted_cigar->positions[cur_index] = pos++;
          converted_cigar->insert_offsets[cur_index] = 0;
          cur_index++;
        }
        break;
      case BAM_CINS: /* I = insertion to the reference */
        /* position stays the same as for the base before, but the insert_offset
         * is increased */
        for (j = 0; j < bam_cigar_oplen(*cigar); j++) {
          // position already points at next spot for next match or deletion,
          // but insertion is present at last position
          converted_cigar->positions[cur_index] = pos - 1;
          converted_cigar->insert_offsets[cur_index] =
              converted_cigar->insert_offsets[cur_index - 1] + 1;
          cur_index++;
        }
        break;
      case BAM_CDEL: /* D = deletion from the reference */
        /* length stores the number of deleted bases in the read according to
     * the reference;
     * to avoid biased analysis results add an 'X' base for every deleted base
     * with averageQuality */
        for (j = 0; j < bam_cigar_oplen(*cigar); j++) {
          converted_cigar->deleted_positions[cur_del_index] = pos++;
          cur_del_index++;
        }
        break;
      case BAM_CSOFT_CLIP:
        /* S = clip on the read with clipped sequence present in qseq */
        /* soft clipping means there were bases that do not match the
     * reference and these are still in the reference BUT only at
     * the start or end of the read -> we ignore these bases and
     * therefore have to skip these bases */
        /* according to sam format specification:
     * S may only have H operations between them and the ends of the CIGAR
     * string
     * -> so if S occurs it must be at the first(0), second (1), (last - 1) or
     * last position in the CIGAR string
     * -> first and second influence the start offset, (last - 1) and last the
     * end offset
     *  */
        if (i == 0 || (i == 1 && !hard_clip_not_present)) {
          /* clipped at the start */
          converted_cigar->start_offset = bam_cigar_oplen(*cigar);
        } else if (i >= (cigarLength - 2)) {
          /* clipped at the end */
          converted_cigar->end_offset = bam_cigar_oplen(*cigar);
        } else {
          /* TODO log it seems not to be standard conform */
        }
        break;
      case BAM_CREF_SKIP: /* N = skip on the reference (e.g. spliced alignment)
                             */
      /* according to the SAM Format reference, this CIGAR operation is only
   * defined
   * for mRNA-to-genome mapping! */
      /* according to
   * http://davetang.org/wiki/tiki-index.php?page=SAM#Spliced_alignment it can
   * be
   * handled like a deletion operator without inserting artificial 'X' bases */
      /* TODO just log it */
      case BAM_CHARD_CLIP: /* H = clip on the read with clipped sequence trimmed
                              off */
        /* hard clipping means there were bases that do not match the reference
     * but they are not in the
     * sequence anymore -> nothing to do just ignore this */
        hard_clip_not_present = false;
        break;
      /* TODO just log it */
      case BAM_CPAD: /* P = padding */
        /* we can ignore the padding operation as it tells us that there is an
     * deletion in the read and
     * also the reference in accordance to another read that inserts bases at
     * this position,
     * see http://davetang.org/wiki/tiki-index.php?page=SAM#Padded_alignment */
        /* TODO just log it */
        break;
        /* BAM_CBACK is not available in all samtools versions */
        /* case BAM_CBACK: */
        /* currently not in the format reference but in the source code */
        /* TODO just log it */
        /* break; */
    }
    /*fprintf(stdout, "%d - %d\n", bam_cigar_op(*cigar),
 bam_cigar_oplen(*cigar));*/
    /* next cigar operator */
    cigar += 1;
  }
  converted_cigar->del_end_offset = static_cast<int>(cur_del_index);
}

// Returns human readable time and according unit string
std::pair<double, std::string>
GenomeDataImporter::_getTimeAsHumanReadableString(
    Timestamp time_in_nanoseconds) {
  int unit = 0;
  double time = time_in_nanoseconds;
  const string units[6] = {"ns", "us", "ms", "s", "min", "h"};
  while (time >= 1000 && unit < 3) {
    time /= 1000;
    unit++;
  }
  while (time >= 60 && unit >= 3 && unit < 5) {
    time /= 60;
    unit++;
  }
  return std::make_pair(time, units[unit]);
}
}

#endif
