#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 28
#define N_INPUT_2_1 28
#define N_INPUT_3_1 1
#define OUT_HEIGHT_2 24
#define OUT_WIDTH_2 24
#define N_FILT_2 6
#define OUT_HEIGHT_4 12
#define OUT_WIDTH_4 12
#define N_FILT_4 6
#define OUT_HEIGHT_5 8
#define OUT_WIDTH_5 8
#define N_FILT_5 8
#define OUT_HEIGHT_7 4
#define OUT_WIDTH_7 4
#define N_FILT_7 8
#define N_SIZE_1_8 128
#define N_LAYER_9 120
#define N_LAYER_11 84
#define N_LAYER_13 10

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<7,3> accum_default_t;
typedef ap_ufixed<1,1,AP_RND_ZERO,AP_SAT> input_t;
typedef ap_fixed<5,1> conv2d_accum_t;
typedef ap_ufixed<1,1,AP_RND_ZERO,AP_SAT> layer2_t;
typedef ap_fixed<2,-1> conv2d_weight_t;
typedef ap_uint<1> bias2_t;
typedef ap_ufixed<1,1,AP_RND_ZERO,AP_SAT> layer4_t;
typedef ap_fixed<8,4> conv2d_1_accum_t;
typedef ap_ufixed<1,1,AP_RND_ZERO,AP_SAT> layer5_t;
typedef ap_fixed<2,-2> weight_default_t;
typedef ap_uint<1> bias5_t;
typedef ap_ufixed<1,1,AP_RND_ZERO,AP_SAT> layer7_t;
typedef ap_ufixed<1,1,AP_RND_ZERO,AP_SAT> layer9_t;
typedef ap_uint<1> bias9_t;
typedef ap_ufixed<1,1,AP_RND_ZERO,AP_SAT> layer11_t;
typedef ap_uint<1> bias11_t;
typedef ap_fixed<7,3> layer13_t;
typedef ap_uint<1> bias13_t;
typedef ap_ufixed<1,1,AP_RND_ZERO,AP_SAT> result_t;

#endif
