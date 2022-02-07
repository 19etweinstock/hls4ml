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

    input_t layer1_out[OUT_HEIGHT_1][OUT_WIDTH_1][N_FILT_1];
    #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    nnet::conv_2d<input_t, input_t, config1>(data, layer1_out, w1, b1);

    input_t layer2_out[OUT_HEIGHT_2][OUT_WIDTH_2][N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::pooling2d<input_t, config2>(layer1_out, layer2_out);

    input_t layer3_out[OUT_HEIGHT_3][OUT_WIDTH_3][N_FILT_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::conv_2d<input_t, input_t, config3>(layer2_out, layer3_out, w3, b3);

    input_t layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    input_t pool2d_layer4_out[OUT_HEIGHT_4][OUT_WIDTH_4][N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=pool2d_layer4_out complete dim=0
    nnet::pooling2d<input_t, config4>(layer3_out, pool2d_layer4_out);
    nnet::flatten<input_t, OUT_HEIGHT_4, OUT_WIDTH_4, N_FILT_4>(pool2d_layer4_out, layer4_out);

    input_t layer5_out[N_LAYER_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    compute_layer5(layer4_out, layer5_out);

    input_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    compute_layer6(layer5_out, layer6_out);

    nnet::compute_layer<input_t, result_t, config7>(layer6_out, res, w7, b7);


}

void compute_layer5(input_t layer4_out[N_LAYER_4], input_t logits5[N_LAYER_5]) {
    input_t logits5_0[32];
    #pragma HLS ARRAY_PARTITION variable=logits5_0 complete dim=0
    input_t logits5_1[32];
    #pragma HLS ARRAY_PARTITION variable=logits5_1 complete dim=0
    input_t logits5_2[32];
    #pragma HLS ARRAY_PARTITION variable=logits5_2 complete dim=0
    input_t logits5_3[24];
    #pragma HLS ARRAY_PARTITION variable=logits5_3 complete dim=0
    input_t logits5_0to1[64];
    #pragma HLS ARRAY_PARTITION variable=logits5_0to1 complete dim=0
    input_t logits5_0to2[96];
    #pragma HLS ARRAY_PARTITION variable=logits5_0to2 complete dim=0
    nnet::compute_layer<input_t, input_t, config5_0>(layer4_out, logits5_0, w5_0, b5_0);
    nnet::compute_layer<input_t, input_t, config5_1>(layer4_out, logits5_1, w5_1, b5_1);
    nnet::compute_layer<input_t, input_t, config5_2>(layer4_out, logits5_2, w5_2, b5_2);
    nnet::compute_layer<input_t, input_t, config5_3>(layer4_out, logits5_3, w5_3, b5_3);
    nnet::merge<input_t, 32, 32>(logits5_0, logits5_1, logits5_0to1);
    nnet::merge<input_t, 64, 32>(logits5_0to1, logits5_2, logits5_0to2);
    nnet::merge<input_t, 96, 24>(logits5_0to2, logits5_3, logits5);
}


void compute_layer6(input_t layer5_out[N_LAYER_5], input_t logits6[N_LAYER_6]) {
    input_t logits6_0[34];
    #pragma HLS ARRAY_PARTITION variable=logits6_0 complete dim=0
    input_t logits6_1[34];
    #pragma HLS ARRAY_PARTITION variable=logits6_1 complete dim=0
    input_t logits6_2[16];
    #pragma HLS ARRAY_PARTITION variable=logits6_2 complete dim=0
    input_t logits6_0to1[68];
    #pragma HLS ARRAY_PARTITION variable=logits6_0to1 complete dim=0
    nnet::compute_layer<input_t, input_t, config6_0>(layer5_out, logits6_0, w6_0, b6_0);
    nnet::compute_layer<input_t, input_t, config6_1>(layer5_out, logits6_1, w6_1, b6_1);
    nnet::compute_layer<input_t, input_t, config6_2>(layer5_out, logits6_2, w6_2, b6_2);
    nnet::merge<input_t, 34, 34>(logits6_0, logits6_1, logits6_0to1);
    nnet::merge<input_t, 68, 16>(logits6_0to1, logits6_2, logits6);
}

