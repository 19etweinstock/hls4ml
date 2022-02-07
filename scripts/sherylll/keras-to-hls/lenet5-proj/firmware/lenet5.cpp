//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include "lenet5.h"

#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_conv2d.h"
#include "nnet_batchnorm.h"
#include "nnet_activation.h"
#include "nnet_pooling.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/w3.h"
#include "weights/b3.h"
#include "weights/w5_0.h"
#include "weights/b5_0.h"
#include "weights/w5_1.h"
#include "weights/b5_1.h"
#include "weights/w5_2.h"
#include "weights/b5_2.h"
#include "weights/w5_3.h"
#include "weights/b5_3.h"
#include "weights/w6_0.h"
#include "weights/b6_0.h"
#include "weights/w6_1.h"
#include "weights/b6_1.h"
#include "weights/w6_2.h"
#include "weights/b6_2.h"
#include "weights/w7.h"
#include "weights/b7.h"

void lenet5(
		  input_t data[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1],
		  result_t res[N_OUTPUTS],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 
    #pragma HLS INTERFACE ap_vld port=data,res 
    #pragma HLS PIPELINE 


    const_size_in   = IN_HEIGHT_1*IN_WIDTH_1*N_CHAN_1;
    const_size_out  = N_OUTPUTS;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer1_t layer1_out[OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    layer1_t conv2d_layer1_out[OUT_HEIGHT_1][OUT_WIDTH_1][N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer1_out complete dim=0
    nnet::conv_2d<input_t, layer1_t, config1>(data, conv2d_layer1_out, w1, b1);
    layer1_t logits1[OUT_HEIGHT_1*OUT_WIDTH_1*N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
    nnet::flatten<layer1_t, OUT_HEIGHT_1, OUT_WIDTH_1, N_FILT_1>(conv2d_layer1_out, logits1);
    nnet::relu<layer1_t, layer1_t, relu_config1>(logits1, layer1_out);

    layer2_t layer2_out[OUT_HEIGHT_2][OUT_WIDTH_2][N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    layer2_t pool2d_layer2_in[IN_HEIGHT_2][IN_WIDTH_2][N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=pool2d_layer2_in complete dim=0
    nnet::unflatten<layer1_t, IN_HEIGHT_2, IN_WIDTH_2, N_FILT_2>(layer1_out, pool2d_layer2_in);
    nnet::pooling2d<layer1_t, config2>(pool2d_layer2_in, layer2_out);

    layer3_t layer3_out[OUT_HEIGHT_3*OUT_WIDTH_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    layer3_t conv2d_layer3_out[OUT_HEIGHT_3][OUT_WIDTH_3][N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=conv2d_layer3_out complete dim=0
    nnet::conv_2d<layer2_t, layer3_t, config3>(layer2_out, conv2d_layer3_out, w3, b3);
    layer3_t logits3[OUT_HEIGHT_3*OUT_WIDTH_3*N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=logits3 complete dim=0
    nnet::flatten<layer3_t, OUT_HEIGHT_3, OUT_WIDTH_3, N_FILT_3>(conv2d_layer3_out, logits3);
    nnet::relu<layer3_t, layer3_t, relu_config3>(logits3, layer3_out);

    layer3_t layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    layer3_t pool2d_layer4_in[IN_HEIGHT_4][IN_WIDTH_4][N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=pool2d_layer4_in complete dim=0
    nnet::unflatten<layer3_t, IN_HEIGHT_4, IN_WIDTH_4, N_FILT_4>(layer3_out, pool2d_layer4_in);
    layer3_t pool2d_layer4_out[OUT_HEIGHT_4][OUT_WIDTH_4][N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=pool2d_layer4_out complete dim=0
    nnet::pooling2d<layer3_t, config4>(pool2d_layer4_in, pool2d_layer4_out);
    nnet::flatten<layer3_t, OUT_HEIGHT_4, OUT_WIDTH_4, N_FILT_4>(pool2d_layer4_out, layer4_out);

    layer5_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    layer5_t logits5[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=logits5 complete dim=0
    compute_layer5(layer4_out, logits5);
    nnet::relu<layer5_t, layer5_t, relu_config5>(logits5, layer5_out);

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    layer6_t logits6[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=logits6 complete dim=0
    compute_layer6(layer5_out, logits6);
    nnet::relu<layer6_t, layer6_t, relu_config6>(logits6, layer6_out);

    result_t logits7[N_OUTPUTS];
    #pragma HLS ARRAY_PARTITION variable=logits7 complete dim=0
    nnet::compute_layer<layer6_t, result_t, config7>(layer6_out, logits7, w7, b7);
    nnet::relu<result_t, result_t, relu_config7>(logits7, res);


}

void compute_layer5(layer4_t layer4_out[N_LAYER_4], layer5_t logits5[N_LAYER_5]) {
    layer5_t logits5_0[32];
    #pragma HLS ARRAY_PARTITION variable=logits5_0 complete dim=0
    layer5_t logits5_1[32];
    #pragma HLS ARRAY_PARTITION variable=logits5_1 complete dim=0
    layer5_t logits5_2[32];
    #pragma HLS ARRAY_PARTITION variable=logits5_2 complete dim=0
    layer5_t logits5_3[24];
    #pragma HLS ARRAY_PARTITION variable=logits5_3 complete dim=0
    layer5_t logits5_0to1[64];
    #pragma HLS ARRAY_PARTITION variable=logits5_0to1 complete dim=0
    layer5_t logits5_0to2[96];
    #pragma HLS ARRAY_PARTITION variable=logits5_0to2 complete dim=0
    nnet::compute_layer<layer4_t, layer5_t, config5_0>(layer4_out, logits5_0, w5_0, b5_0);
    nnet::compute_layer<layer4_t, layer5_t, config5_1>(layer4_out, logits5_1, w5_1, b5_1);
    nnet::compute_layer<layer4_t, layer5_t, config5_2>(layer4_out, logits5_2, w5_2, b5_2);
    nnet::compute_layer<layer4_t, layer5_t, config5_3>(layer4_out, logits5_3, w5_3, b5_3);
    nnet::merge<layer5_t, 32, 32>(logits5_0, logits5_1, logits5_0to1);
    nnet::merge<layer5_t, 64, 32>(logits5_0to1, logits5_2, logits5_0to2);
    nnet::merge<layer5_t, 96, 24>(logits5_0to2, logits5_3, logits5);
}


void compute_layer6(layer5_t layer5_out[N_LAYER_5], layer6_t logits6[N_LAYER_6]) {
    layer6_t logits6_0[34];
    #pragma HLS ARRAY_PARTITION variable=logits6_0 complete dim=0
    layer6_t logits6_1[34];
    #pragma HLS ARRAY_PARTITION variable=logits6_1 complete dim=0
    layer6_t logits6_2[16];
    #pragma HLS ARRAY_PARTITION variable=logits6_2 complete dim=0
    layer6_t logits6_0to1[68];
    #pragma HLS ARRAY_PARTITION variable=logits6_0to1 complete dim=0
    nnet::compute_layer<layer5_t, layer6_t, config6_0>(layer5_out, logits6_0, w6_0, b6_0);
    nnet::compute_layer<layer5_t, layer6_t, config6_1>(layer5_out, logits6_1, w6_1, b6_1);
    nnet::compute_layer<layer5_t, layer6_t, config6_2>(layer5_out, logits6_2, w6_2, b6_2);
    nnet::merge<layer6_t, 34, 34>(logits6_0, logits6_1, logits6_0to1);
    nnet::merge<layer6_t, 68, 16>(logits6_0to1, logits6_2, logits6);
}

