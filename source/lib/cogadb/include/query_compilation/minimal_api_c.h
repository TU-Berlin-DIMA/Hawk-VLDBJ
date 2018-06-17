
#ifndef MINIMAL_API_H
#define MINIMAL_API_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <hardware_optimizations/main_memory_joins/serial_hash_join/hashtable/hashtable.h>

#ifndef __clang__
typedef uint64_t size_t;
#endif
typedef uint64_t TID;

struct C_Table;
typedef struct C_Table C_Table;

struct C_Column;
typedef struct C_Column C_Column;

struct C_HashTable;
typedef struct C_HashTable C_HashTable;

typedef void (*HashTableCleanupFunctionPtr)(void*);

struct C_State;
typedef struct C_State C_State;

C_Table* getTableByName(const char* const column_name);
C_Column* getColumnByName(C_Table*, const char* const column_name);
C_Column* getColumnById(C_Table*, const unsigned int id);
C_HashTable* getHashTable(C_Table*, const char* const column_name);

void releaseTable(C_Table*);
void releaseColumn(C_Column*);
void releaseHashTable(C_HashTable*);

const int32_t* getArrayFromColumn_int32_t(C_Column*);
const uint32_t* getArrayFromColumn_uint32_t(C_Column*);
const uint64_t* getArrayFromColumn_uint64_t(C_Column*);
const float* getArrayFromColumn_float(C_Column*);
const double* getArrayFromColumn_double(C_Column*);
const char* const* getArrayFromColumn_cstring(C_Column*);
const char* getArrayFromColumn_char(C_Column*);

uint32_t* getArrayCompressedKeysFromColumn_string(C_Column*);

C_Table* createTableFromColumns(const char* const table_name,
                                C_Column** columns, size_t num_columns);
bool addHashTable(C_Table*, const char* const column_name,
                  C_HashTable* hash_table);

C_HashTable* createSystemHashTable(void*, HashTableCleanupFunctionPtr, char*);
void* getHashTableFromSystemHashTable(C_HashTable*);

size_t getNumberOfRows(C_Table*);

C_Column* createResultArray_int32_t(const char* const name,
                                    const int32_t* array, size_t num_elements);
C_Column* createResultArray_uint32_t(const char* const name,
                                     const uint32_t* array,
                                     size_t num_elements);
C_Column* createResultArray_uint64_t(const char* const name,
                                     const uint64_t* array,
                                     size_t num_elements);
C_Column* createResultArray_float(const char* const name, const float* array,
                                  size_t num_elements);
C_Column* createResultArray_double(const char* const name, const double* array,
                                   size_t num_elements);
C_Column* createResultArray_cstring(const char* const name, const char** array,
                                    size_t num_elements);
C_Column* createResultArray_char(const char* const name, const char* array,
                                 size_t num_elements);

C_Column* createResultArray_cstring_compressed(C_Column* orig,
                                               const char* const name,
                                               const uint32_t* array,
                                               size_t num_elements);

char** stringMalloc(size_t number_of_elements, size_t max_length);
void stringFree(char** data, size_t number_of_elements);
char** stringRealloc(char** data, size_t old_number_of_elements,
                     size_t number_of_elements, size_t max_length);

size_t getMaxStringLengthForColumn(C_Column*);

uint64_t getGroupKey(unsigned int count, ...);

struct C_AggregationHashTable;
typedef struct C_AggregationHashTable C_AggregationHashTable;
struct C_AggregationHashTableIterator;
typedef struct C_AggregationHashTableIterator C_AggregationHashTableIterator;

C_AggregationHashTable* createAggregationHashTable(uint64_t payload_size);
void freeAggregationHashTable(C_AggregationHashTable*);
uint64_t getAggregationHashTableSize(C_AggregationHashTable*);
void* getAggregationHashTablePayload(C_AggregationHashTable*,
                                     uint64_t group_key);
void* insertAggregationHashTable(C_AggregationHashTable*, uint64_t group_key,
                                 void* payload);

C_AggregationHashTableIterator* createAggregationHashTableIterator(
    C_AggregationHashTable*);
void freeAggregationHashTableIterator(C_AggregationHashTableIterator*);
void nextAggregationHashTableIterator(C_AggregationHashTableIterator*);
char hasNextAggregationHashTableIterator(C_AggregationHashTableIterator*);
void* getAggregationHashTableIteratorPayload(C_AggregationHashTableIterator*);

struct C_UniqueKeyHashTable;
typedef struct C_UniqueKeyHashTable C_UniqueKeyHashTable;

C_UniqueKeyHashTable* createUniqueKeyJoinHashTable(uint64_t num_elements);
void freeUniqueKeyJoinHashTable(C_UniqueKeyHashTable*);
uint64_t getUniqueKeyJoinHashTableSize(C_UniqueKeyHashTable*);
uint64_t* getUniqueKeyJoinHashTablePayload(C_UniqueKeyHashTable*, uint64_t key);
int insertUniqueKeyJoinHashTable(C_UniqueKeyHashTable*, uint64_t key,
                                 uint64_t payload);

#define C_MIN(a, b) (a = (a < b ? a : b))
#define C_MIN_uint64t(a, b) C_MIN(a, b)
#define C_MIN_double(a, b) C_MIN(a, b)

#define C_MAX(a, b) (a = (a > b ? a : b))
#define C_MAX_uint64_t(a, b) C_MAX(a, b)
#define C_MAX_double(a, b) C_MAX(a, b)

#define C_SUM(a, b) (a += b)
#define C_SUM_uint64_t(a, b) C_SUM(a, b)
#define C_SUM_double(a, b) C_SUM(a, b)

#ifdef __cplusplus
}
#endif

#endif /* MINIMAL_API_H */
