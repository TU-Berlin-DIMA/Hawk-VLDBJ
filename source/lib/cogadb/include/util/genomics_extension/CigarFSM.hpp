/*
 * File:   CigarFSM.hpp
 * Author: John Sarrazin
 *
 * Created on 25. Juni 2015, 11:20
 *
 * The CigarFSM takes several types of Input to calculate the Cigar String for
 * the current read. Use the function reset() if you want to start with a new
 * Read.
 */
#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

class CigarFSM {
 public:
  /**
   * \brief Default Constructor to create a CigarFSM.
   * The Constructor creates a simple FSM with the initial setup.
   */
  CigarFSM();

  /**
   * \brief Default Destructor.
   */
  ~CigarFSM();

  /**
   * \brief Handles Input by char variables.
   * Handles Inputs by char values.
   * @param input i or I for Insert, d or D for Delete, m or M for Match.
   */
  //    void handleInput(char input);

  // void handleInput(istream& input);
  /**
   * \brief Resets the Final State Mashine.
   * Sets all states to initial Values. Starts a new sequence for cigar
   * calculation.
   */
  void reset();

  /**
   * \brief Returns a Reference to the Cigar-String calculated so far.
   * Uses the current state of the State Mashine to calculate the Cigar-String
   * for the current Input sequence. Does not Reset the State Mashine.
   * @return Reference to the Cigar-String calculated so far.
   */
  const std::string getCigar();

  /**
   * \brief Handles Insertion Input
   * Handles the case, when there is an Insertion at current position of Sample
   * Genome
   * in comparison to the Reference Genome.
   */
  void handleInsert();

  /**
   * \brief Handles Deletion Input
   * Handles the case, when there is an Deletion at current position of Sample
   * Genome
   * in comparison to the Reference Genome.
   */
  void handleDelete();

  /**
   * \brief Handles Match Input
   * Handles the case, when the Sample Base matches the Reference Base
   */
  void handleMatch();

  /**
   * \brief Handles data stored in the Database as Input.
   * Calculates the Input type by using the Data that is stored in the
   * genomic extension of CogaDB.
   * @param baseValue sb_base_value of current position.
   * @param insertOffset sb_insert_offset of current position.
   */
  void handleDatabaseInput(std::string baseValue, int insertOffset);

 private:
  // Represents the different states of the State Mashine.
  enum State { START, INSERT, DELETE, MATCH };

  // Represents the state the State Mashine is currently.
  State currentState;

  // Counts the insertions in the current subsequence. Resets, if the state
  // changes.
  unsigned int insertCount;

  // Counts the matches in the current subsequence. Resets, if the state
  // changes.
  unsigned int matchCount;

  // Counts the deletions in the current subsequence. Resets, if the state
  // changes.
  unsigned int deleteCount;

  // represents the cigar string after the last state change.
  std::string cigar;
};
