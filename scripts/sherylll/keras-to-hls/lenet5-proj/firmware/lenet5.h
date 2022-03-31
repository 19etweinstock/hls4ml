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

#ifndef LENET5_H_
#define LENET5_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "parameters.h"


// Prototype of top level function for C-synthesis
void lenet5(
      input_t* input,
      result_t* zero, result_t* one, result_t* two, result_t* three, result_t* four,
      result_t* five, result_t* six, result_t* seven, result_t* eight, result_t* nine,
      max_t* max, last_layer_t* last_layer, one_count_t* one_count);

void compute_layer5(layer_t layer4_out[N_LAYER_4], layer_t logits5[N_LAYER_5]);
void compute_layer6(layer_t layer5_out[N_LAYER_5], layer_t logits6[N_LAYER_6]);

#endif

