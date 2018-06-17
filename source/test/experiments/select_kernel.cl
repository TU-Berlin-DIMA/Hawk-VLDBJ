typedef unsigned char		uint8_t;
typedef unsigned short int	uint16_t;
typedef unsigned int		uint32_t;
typedef unsigned long int	uint64_t;

typedef char		int8_t;
typedef short int	int16_t;
typedef int		int32_t;
typedef long int	int64_t;

/*
__kernel void select_and_hashprobe_kernel(uint64_t num_elements_tuple_id_LINEORDER1, 
    __global const int32_t * array_LINEORDER1_LINEORDER_LO_QUANTITY1,
    __global const float * array_LINEORDER1_LINEORDER_LO_DISCOUNT1,
    __global const float * array_LINEORDER1_LINEORDER_LO_REVENUE1,
    __global char* flags){

    uint64_t tuple_id_LINEORDER1=get_global_id(0);
    if(((array_LINEORDER1_LINEORDER_LO_REVENUE1[tuple_id_LINEORDER1] > 4900000.0f) && (array_LINEORDER1_LINEORDER_LO_DISCOUNT1[tuple_id_LINEORDER1] >= 1.0f) && (array_LINEORDER1_LINEORDER_LO_DISCOUNT1[tuple_id_LINEORDER1] <= 3.0f) && (array_LINEORDER1_LINEORDER_LO_QUANTITY1[tuple_id_LINEORDER1] < 25))) {
        flags[tuple_id_LINEORDER1]=1;
    }else{
        flags[tuple_id_LINEORDER1]=0;
    }
}

__kernel void hashprobe_aggregate_and_project_kernel(uint64_t num_elements_tuple_id_LINEORDER1, 
    __global const char* flags,
    __global const uint64_t* write_positions,
    __global const int32_t * array_LINEORDER1_LINEORDER_LO_QUANTITY1,
    __global const float * array_LINEORDER1_LINEORDER_LO_DISCOUNT1,
    __global const float * array_LINEORDER1_LINEORDER_LO_REVENUE1,
    __global int32_t* result_array_LINEORDER_LO_QUANTITY_1,
    __global float* result_array_LINEORDER_LO_DISCOUNT_1,
    __global float* result_array_LINEORDER_LO_REVENUE_1){

    uint64_t tuple_id_LINEORDER1=get_global_id(0);
    if(flags[tuple_id_LINEORDER1]){
        uint64_t write_pos = write_positions[tuple_id_LINEORDER1];
        result_array_LINEORDER_LO_QUANTITY_1[write_pos] = array_LINEORDER1_LINEORDER_LO_QUANTITY1[tuple_id_LINEORDER1];
        result_array_LINEORDER_LO_DISCOUNT_1[write_pos] = array_LINEORDER1_LINEORDER_LO_DISCOUNT1[tuple_id_LINEORDER1];
        result_array_LINEORDER_LO_REVENUE_1[write_pos] = array_LINEORDER1_LINEORDER_LO_REVENUE1[tuple_id_LINEORDER1];
    }
}
*/

__kernel void select_and_hashprobe_kernel(uint64_t num_elements_tuple_id_LINEORDER1, 
    __global const int32_t * array_LINEORDER1_LINEORDER_LO_QUANTITY1,
    __global const float * array_LINEORDER1_LINEORDER_LO_DISCOUNT1,
    __global const float * array_LINEORDER1_LINEORDER_LO_REVENUE1,
    __global char* flags){

    uint64_t tuple_id_LINEORDER1=(num_elements_tuple_id_LINEORDER1/4)*get_global_id(0);
    uint64_t tmp =  tuple_id_LINEORDER1+(num_elements_tuple_id_LINEORDER1/4);
    uint64_t end_index;
    if(num_elements_tuple_id_LINEORDER1 > tmp){
        end_index = tmp;
    }else{
        end_index = num_elements_tuple_id_LINEORDER1;
    }
    
    for(;tuple_id_LINEORDER1<end_index;++tuple_id_LINEORDER1){
        if(((array_LINEORDER1_LINEORDER_LO_REVENUE1[tuple_id_LINEORDER1] > 4900000.0f) 
        && (array_LINEORDER1_LINEORDER_LO_DISCOUNT1[tuple_id_LINEORDER1] >= 1.0f) 
        && (array_LINEORDER1_LINEORDER_LO_DISCOUNT1[tuple_id_LINEORDER1] <= 3.0f) 
        && (array_LINEORDER1_LINEORDER_LO_QUANTITY1[tuple_id_LINEORDER1] < 25))) {
            flags[tuple_id_LINEORDER1]=1;
        }else{
            flags[tuple_id_LINEORDER1]=0;
        }
    }
}

__kernel void hashprobe_aggregate_and_project_kernel(uint64_t num_elements_tuple_id_LINEORDER1, 
    __global const char* flags,
    __global const uint64_t* write_positions,
    __global const int32_t * array_LINEORDER1_LINEORDER_LO_QUANTITY1,
    __global const float * array_LINEORDER1_LINEORDER_LO_DISCOUNT1,
    __global const float * array_LINEORDER1_LINEORDER_LO_REVENUE1,
    __global int32_t* result_array_LINEORDER_LO_QUANTITY_1,
    __global float* result_array_LINEORDER_LO_DISCOUNT_1,
    __global float* result_array_LINEORDER_LO_REVENUE_1){

    uint64_t tuple_id_LINEORDER1=(num_elements_tuple_id_LINEORDER1/4)*get_global_id(0);
    uint64_t tmp =  tuple_id_LINEORDER1+(num_elements_tuple_id_LINEORDER1/4);
    uint64_t end_index;
    if(num_elements_tuple_id_LINEORDER1 > tmp){
        end_index = tmp;
    }else{
        end_index = num_elements_tuple_id_LINEORDER1;
    }
    
    for(;tuple_id_LINEORDER1<end_index;++tuple_id_LINEORDER1){
        if(flags[tuple_id_LINEORDER1]){
            uint64_t write_pos = write_positions[tuple_id_LINEORDER1];
            result_array_LINEORDER_LO_QUANTITY_1[write_pos] = array_LINEORDER1_LINEORDER_LO_QUANTITY1[tuple_id_LINEORDER1];
            result_array_LINEORDER_LO_DISCOUNT_1[write_pos] = array_LINEORDER1_LINEORDER_LO_DISCOUNT1[tuple_id_LINEORDER1];
            result_array_LINEORDER_LO_REVENUE_1[write_pos] = array_LINEORDER1_LINEORDER_LO_REVENUE1[tuple_id_LINEORDER1];
        }
    }
}