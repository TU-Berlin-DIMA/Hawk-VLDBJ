//
// Created by basti on 30.06.15.
//

//#pragma once

#ifndef GENOME_ALIGNMENT_FILES_HPP
#define GENOME_ALIGNMENT_FILES_HPP

#ifdef BAM_FOUND

#include <samtools/bam.h>
#include <samtools/sam.h>
#include <core/global_definitions.hpp>

namespace CoGaDB {
  using namespace std;

  class AlignmentFile {
   protected:
    string path_to_alignment_file;

   public:
    AlignmentFile(string);

    virtual ~AlignmentFile();

    virtual void openFile() = 0;

    virtual int getNextAlignment(bam1_t *alignment) = 0;

    virtual bam_header_t *getHeader() = 0;

    virtual void closeFile() = 0;
  };

  class BAMFile : public AlignmentFile {
   private:
    bamFile input_file;
    bam_header_t *header;

   public:
    BAMFile(string);

    void openFile();

    int getNextAlignment(bam1_t *);

    bam_header_t *getHeader();

    void closeFile();
  };

  class SAMFile : public AlignmentFile {
   private:
    tamFile input_file;
    bam_header_t *header;

   public:
    SAMFile(string);

    void openFile();

    int getNextAlignment(bam1_t *);

    bam_header_t *getHeader();

    void closeFile();
  };

}  // end namespace CoGaDB

#endif

#endif  // GENOME_ALIGNMENT_FILES_HPP
