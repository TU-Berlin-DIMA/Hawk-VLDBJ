//
// Created by Sebastian Dorok on 30.06.15.
//

#include <core/global_definitions.hpp>

#ifdef BAM_FOUND

#include <util/genomics_extension/alignment_files.hpp>

namespace CoGaDB {

using namespace std;

AlignmentFile::AlignmentFile(string _path_to_alignment_file)
    : path_to_alignment_file(_path_to_alignment_file) {}

AlignmentFile::~AlignmentFile() {}

BAMFile::BAMFile(string _path_to_alignment_file)
    : AlignmentFile(_path_to_alignment_file) {}

void BAMFile::openFile() {
  input_file = bam_open(path_to_alignment_file.c_str(), "r");
  header = bam_header_read(input_file);
}

int BAMFile::getNextAlignment(bam1_t *alignment) {
  return bam_read1(input_file, alignment);
}

bam_header_t *BAMFile::getHeader() {
  assert(header != NULL);
  return header;
}

void BAMFile::closeFile() {
  bam_header_destroy(header);
  bam_close(input_file);
}

SAMFile::SAMFile(string _path_to_alignment_file)
    : AlignmentFile(_path_to_alignment_file) {}

void SAMFile::openFile() {
  input_file = sam_open(path_to_alignment_file.c_str());
  header = sam_header_read(input_file);
}

int SAMFile::getNextAlignment(bam1_t *alignment) {
  return sam_read1(input_file, header, alignment);
}

bam_header_t *SAMFile::getHeader() {
  assert(header != NULL);
  return header;
}

void SAMFile::closeFile() {
  bam_header_destroy(header);
  sam_close(input_file);
}

}  // end namespace CoGaDB

#endif
