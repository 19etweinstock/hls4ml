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
		  input_t* input,
		  result_t* zero, result_t* one, result_t* two, result_t* three, result_t* four,
          result_t* five, result_t* six, result_t* seven, result_t* eight, result_t* nine,
          result_t* max
        )
{

    //hls-fpga-machine-learning insert IO
    // #pragma HLS ARRAY_RESHAPE variable=data complete dim=0 
    // #pragma HLS ARRAY_RESHAPE variable=res complete dim=0 
    // #pragma HLS INTERFACE ap_bus port=data
    // #pragma HLS INTERFACE ap_stable port=res
    // #pragma HLS PIPELINE 

    layer_t data[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1];

    for (int i = 0; i < IN_HEIGHT_1; i++)
        for (int j = 0; j < IN_WIDTH_1; j++){
            data[i][j][0] = (*input >> (i * IN_HEIGHT_1 + j)) & 0x1;
        }


    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer_t layer1_out[OUT_HEIGHT_1][OUT_WIDTH_1][N_FILT_1];
    // // #pragma HLS ARRAY_PARTITION variable=layer1_out complete dim=0
    nnet::conv_2d_resource_cl<layer_t, layer_t, config1, config1_mult>(data, layer1_out, w1, b1);

    layer_t layer2_out[OUT_HEIGHT_2][OUT_WIDTH_2][N_FILT_2];
    // // #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::pooling2d<layer_t, config2>(layer1_out, layer2_out);

    layer_t layer3_out[OUT_HEIGHT_3][OUT_WIDTH_3][N_FILT_3];
    // // #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::conv_2d_resource_cl<layer_t, layer_t, config3, config3_mult>(layer2_out, layer3_out, w3, b3);

    layer_t layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    // // #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::pooling2d_flatten<layer_t, config4>(layer3_out, layer4_out);

    layer_t layer5_out[N_LAYER_5];
    // // #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    compute_layer5(layer4_out, layer5_out);

    layer_t layer6_out[N_LAYER_6];
    // // #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    compute_layer6(layer5_out, layer6_out);

    result_t res[N_OUTPUTS];
    nnet::compute_layer<layer_t, result_t, config7>(layer6_out, res, w7, b7);

    *zero = res[0];
    *one = res[1];
    *two = res[2];
    *three = res[3];
    *four = res[4];
    *five = res[5];
    *six = res[6];
    *seven = res[7];
    *eight = res[8];
    *nine = res[9];

    result_t max_val = res[0];
    *max = 0;
    for (int i = 1; i< 9; i++){
        if (res[i] > max_val){
            max_val = res[i];
            *max = i;
        }
    }


}

void compute_layer5(layer_t layer4_out[N_LAYER_4], layer_t logits5[N_LAYER_5]) {
    // #pragma HLS INLINE
    // layer_t logits5_0[32];
    // // // #pragma HLS ARRAY_PARTITION variable=logits5_0 complete dim=0
    // layer_t logits5_1[32];
    // // // #pragma HLS ARRAY_PARTITION variable=logits5_1 complete dim=0
    // layer_t logits5_2[32];
    // // // #pragma HLS ARRAY_PARTITION variable=logits5_2 complete dim=0
    // layer_t logits5_3[24];
    // // // #pragma HLS ARRAY_PARTITION variable=logits5_3 complete dim=0
    // layer_t logits5_0to1[64];
    // // // #pragma HLS ARRAY_PARTITION variable=logits5_0to1 complete dim=0
    // layer_t logits5_0to2[96];
    // // // #pragma HLS ARRAY_PARTITION variable=logits5_0to2 complete dim=0
    nnet::compute_layer<layer_t, layer_t, config5_0>(layer4_out, logits5, w5_0, b5_0);
    nnet::compute_layer<layer_t, layer_t, config5_1>(layer4_out, &logits5[32], w5_1, b5_1);
    nnet::compute_layer<layer_t, layer_t, config5_2>(layer4_out, &logits5[64], w5_2, b5_2);
    nnet::compute_layer<layer_t, layer_t, config5_3>(layer4_out, &logits5[96], w5_3, b5_3);
    // nnet::merge<layer_t, 32, 32>(logits5_0, logits5_1, logits5_0to1);
    // nnet::merge<layer_t, 64, 32>(logits5_0to1, logits5_2, logits5_0to2);
    // nnet::merge<layer_t, 96, 24>(logits5_0to2, logits5_3, logits5);
}


void compute_layer6(layer_t layer5_out[N_LAYER_5], layer_t logits6[N_LAYER_6]) {
    // #pragma HLS INLINE
    // layer_t logits6_0[34];
    // // // #pragma HLS ARRAY_PARTITION variable=logits6_0 complete dim=0
    // layer_t logits6_1[34];
    // // // #pragma HLS ARRAY_PARTITION variable=logits6_1 complete dim=0
    // layer_t logits6_2[16];
    // // // #pragma HLS ARRAY_PARTITION variable=logits6_2 complete dim=0
    // layer_t logits6_0to1[68];
    // // // #pragma HLS ARRAY_PARTITION variable=logits6_0to1 complete dim=0
    nnet::compute_layer<layer_t, layer_t, config6_0>(layer5_out, logits6, w6_0, b6_0);
    nnet::compute_layer<layer_t, layer_t, config6_1>(layer5_out, &logits6[34], w6_1, b6_1);
    nnet::compute_layer<layer_t, layer_t, config6_2>(layer5_out, &logits6[68], w6_2, b6_2);
    // nnet::merge<layer_t, 34, 34>(logits6_0, logits6_1, logits6_0to1);
    // nnet::merge<layer_t, 68, 16>(logits6_0to1, logits6_2, logits6);
}

