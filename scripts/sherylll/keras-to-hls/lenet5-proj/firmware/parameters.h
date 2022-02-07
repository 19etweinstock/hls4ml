#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_batchnorm.h"
#include "nnet_pooling.h"

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<7,3> accum_default_t;
typedef ap_fixed<5,2> accum_default_conv0_t;
typedef ap_fixed<8,4> accum_default_conv1_t;
typedef ap_fixed<2,0> weight_default_layer1_t;
typedef ap_fixed<2,-1> weight_default_t;
typedef ap_fixed<1,0> bias_default_t;
typedef ap_ufixed<1,0, AP_RND_ZERO, AP_SAT> input_t;
typedef ap_fixed<7,3> result_t;
#define IN_HEIGHT_1 28
#define IN_WIDTH_1 28
#define N_CHAN_1 1
#define OUT_HEIGHT_1 24
#define OUT_WIDTH_1 24
#define N_FILT_1 6
#define IN_HEIGHT_2 24
#define IN_WIDTH_2 24
#define OUT_HEIGHT_2 12
#define OUT_WIDTH_2 12
#define POOL_HEIGHT_2 2
#define POOL_WIDTH_2 2
#define N_FILT_2 6
#define N_LAYER_2 864
#define IN_HEIGHT_3 12
#define IN_WIDTH_3 12
#define N_CHAN_3 6
#define OUT_HEIGHT_3 8
#define OUT_WIDTH_3 8
#define N_FILT_3 8
#define IN_HEIGHT_4 8
#define IN_WIDTH_4 8
#define OUT_HEIGHT_4 4
#define OUT_WIDTH_4 4
#define POOL_HEIGHT_4 2
#define POOL_WIDTH_4 2
#define N_FILT_4 8
#define N_LAYER_4 128
#define N_LAYER_5 120
#define N_LAYER_6 84
#define N_OUTPUTS 10

//hls-fpga-machine-learning insert layer-config
struct config1 : nnet::conv2d_config {
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const unsigned in_height = IN_HEIGHT_1;
        static const unsigned in_width = IN_WIDTH_1;
        static const unsigned n_chan = N_CHAN_1;
        static const unsigned filt_height = 5;
        static const unsigned filt_width = 5;
        static const unsigned n_filt = N_FILT_1;
        static const unsigned stride_height = 1;
        static const unsigned stride_width = 1;
        static const unsigned out_height = OUT_HEIGHT_1;
        static const unsigned out_width = OUT_WIDTH_1;
        static const unsigned multiplier_limit = 86400;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_conv0_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_layer1_t weight_t;
        };
struct config2 : nnet::pooling2d_config {
        static const unsigned in_height = IN_HEIGHT_2;
        static const unsigned in_width = IN_WIDTH_2;
        static const unsigned n_filt = N_FILT_2;
        static const unsigned stride_height = 2;
        static const unsigned stride_width = 2;
        static const unsigned pool_height = 2;
        static const unsigned pool_width = 2;
        static const unsigned out_height = OUT_HEIGHT_2;
        static const unsigned out_width = OUT_WIDTH_2;
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const nnet::Pool_Op pool_op = nnet::Max;
        static const unsigned reuse = 1;
    };

    struct config3 : nnet::conv2d_config {
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const unsigned in_height = IN_HEIGHT_3;
        static const unsigned in_width = IN_WIDTH_3;
        static const unsigned n_chan = N_CHAN_3;
        static const unsigned filt_height = 5;
        static const unsigned filt_width = 5;
        static const unsigned n_filt = N_FILT_3;
        static const unsigned stride_height = 1;
        static const unsigned stride_width = 1;
        static const unsigned out_height = OUT_HEIGHT_3;
        static const unsigned out_width = OUT_WIDTH_3;
        static const unsigned multiplier_limit = 172800;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_conv1_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct config4 : nnet::pooling2d_config {
        static const unsigned in_height = IN_HEIGHT_4;
        static const unsigned in_width = IN_WIDTH_4;
        static const unsigned n_filt = N_FILT_4;
        static const unsigned stride_height = 2;
        static const unsigned stride_width = 2;
        static const unsigned pool_height = 2;
        static const unsigned pool_width = 2;
        static const unsigned out_height = OUT_HEIGHT_4;
        static const unsigned out_width = OUT_WIDTH_4;
        static const unsigned pad_top = 0;
        static const unsigned pad_bottom = 0;
        static const unsigned pad_left = 0;
        static const unsigned pad_right = 0;
        static const nnet::Pool_Op pool_op = nnet::Max;
        static const unsigned reuse = 1;
    };

    struct config5_0 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_4;
        static const unsigned n_out = 32;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct config5_1 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_4;
        static const unsigned n_out = 32;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct config5_2 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_4;
        static const unsigned n_out = 32;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct config5_3 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_4;
        static const unsigned n_out = 24;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct config6_0 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_5;
        static const unsigned n_out = 34;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct config6_1 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_5;
        static const unsigned n_out = 34;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct config6_2 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_5;
        static const unsigned n_out = 16;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };
struct config7 : nnet::layer_config {
        static const unsigned n_in = N_LAYER_6;
        static const unsigned n_out = N_OUTPUTS;
        static const unsigned io_type = nnet::io_parallel;
        static const unsigned reuse_factor = 1;
        static const unsigned n_zeros = 0;
        static const bool store_weights_in_bram = false;
        typedef accum_default_t accum_t;
        typedef bias_default_t bias_t;
        typedef weight_default_t weight_t;
        };

#endif 
