#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_pooling.h"
#include "nnet_utils/nnet_pooling_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/w9.h"
#include "weights/b9.h"
#include "weights/w11.h"
#include "weights/b11.h"
#include "weights/w13.h"
#include "weights/b13.h"

//hls-fpga-machine-learning insert layer-config
// conv2d
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 25;
    static const unsigned n_out = 6;
    static const unsigned reuse_factor = 5;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<5,1> accum_t;
    typedef bias2_t bias_t;
    typedef conv2d_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config2 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = N_INPUT_1_1;
    static const unsigned in_width = N_INPUT_2_1;
    static const unsigned n_chan = N_INPUT_3_1;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_2;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_2;
    static const unsigned out_width = OUT_WIDTH_2;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = N_INPUT_1_1;
    static const unsigned min_width = N_INPUT_2_1;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<5,1> accum_t;
    typedef bias2_t bias_t;
    typedef conv2d_weight_t weight_t;
    typedef config2_mult mult_config;
};
const ap_uint<config2::filt_height * config2::filt_width> config2::pixels[] = {0};

// max_pooling2d
struct config4 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_filt = N_FILT_4;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_4;

    static const unsigned out_height = OUT_HEIGHT_4;
    static const unsigned out_width = OUT_WIDTH_4;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 1;
    typedef ap_fixed<7,3> accum_t;
};

// conv2d_1
struct config5_mult : nnet::dense_config {
    static const unsigned n_in = 150;
    static const unsigned n_out = 8;
    static const unsigned reuse_factor = 5;
    static const unsigned strategy = nnet::resource;
    typedef ap_fixed<8,4> accum_t;
    typedef bias5_t bias_t;
    typedef weight_default_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config5 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_4;
    static const unsigned in_width = OUT_WIDTH_4;
    static const unsigned n_chan = N_FILT_4;
    static const unsigned filt_height = 5;
    static const unsigned filt_width = 5;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_5;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_5;
    static const unsigned out_width = OUT_WIDTH_5;
    static const unsigned reuse_factor = 5;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::resource;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = OUT_HEIGHT_4;
    static const unsigned min_width = OUT_WIDTH_4;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<8,4> accum_t;
    typedef bias5_t bias_t;
    typedef weight_default_t weight_t;
    typedef config5_mult mult_config;
};
const ap_uint<config5::filt_height * config5::filt_width> config5::pixels[] = {0};

// max_pooling2d_1
struct config7 : nnet::pooling2d_config {
    static const unsigned in_height = OUT_HEIGHT_5;
    static const unsigned in_width = OUT_WIDTH_5;
    static const unsigned n_filt = N_FILT_7;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned pool_height = 2;
    static const unsigned pool_width = 2;

    static const unsigned filt_height = 2;
    static const unsigned filt_width = 2;
    static const unsigned n_chan = N_FILT_7;

    static const unsigned out_height = OUT_HEIGHT_7;
    static const unsigned out_width = OUT_WIDTH_7;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const nnet::Pool_Op pool_op = nnet::Max;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned reuse = 1;
    typedef ap_fixed<7,3> accum_t;
};

// dense
struct config9 : nnet::dense_config {
    static const unsigned n_in = N_SIZE_1_8;
    static const unsigned n_out = N_LAYER_9;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 15360;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<7,3> accum_t;
    typedef bias9_t bias_t;
    typedef weight_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// dense_1
struct config11 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_9;
    static const unsigned n_out = N_LAYER_11;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 10080;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<7,3> accum_t;
    typedef bias11_t bias_t;
    typedef weight_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// dense_2
struct config13 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_11;
    static const unsigned n_out = N_LAYER_13;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::resource;
    static const unsigned reuse_factor = 2;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 840;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<7,3> accum_t;
    typedef bias13_t bias_t;
    typedef weight_default_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// dense_2_linear
struct linear_config14 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_13;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};


#endif
