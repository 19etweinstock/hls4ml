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
      input_t data[IN_HEIGHT_1][IN_WIDTH_1][N_CHAN_1],
      result_t res[N_OUTPUTS],
      unsigned short &const_size_in,
      unsigned short &const_size_out);

void compute_layer5(input_t layer4_out[N_LAYER_4], input_t logits5[N_LAYER_5]);
void compute_layer6(input_t layer5_out[N_LAYER_5], input_t logits6[N_LAYER_6]);

#endif

