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


// for csynth make sure that nested exception and exception ptr files are not included since they cause erors
#ifdef __clang__
class type_info;
#include <exception_ptr.h>
#define _GLIBCXX_NESTED_EXCEPTION_H 1

#pragma GCC visibility push(default)

#ifndef __GXX_EXPERIMENTAL_CXX0X__
# include <bits/c++0x_warning.h>
#else

#include <bits/c++config.h>

#if !defined(_GLIBCXX_ATOMIC_BUILTINS_4)
#  error This platform does not support exception propagation.
#endif

extern "C++" {

namespace std
{
  /**
   * @addtogroup exceptions
   * @{
   */

  /// Exception class with exception_ptr data member.
  class nested_exception
  {
    exception_ptr _M_ptr;

  public:
    nested_exception() throw() : _M_ptr(current_exception()) { }

    nested_exception(const nested_exception&) = default;

    nested_exception& operator=(const nested_exception&) = default;

    inline virtual ~nested_exception();

    void
    rethrow_nested() const __attribute__ ((__noreturn__))
    { rethrow_exception(_M_ptr); }

    exception_ptr
    nested_ptr() const
    { return _M_ptr; }
  };

  inline nested_exception::~nested_exception() = default;

  template<typename _Except>
    struct _Nested_exception : public _Except, public nested_exception
    {
      explicit _Nested_exception(_Except&& __ex)
      : _Except(static_cast<_Except&&>(__ex))
      { }
    };

  template<typename _Ex>
    struct __get_nested_helper
    {
      static const nested_exception*
      _S_get(const _Ex& __ex)
      { return dynamic_cast<const nested_exception*>(&__ex); }
    };

  template<typename _Ex>
    struct __get_nested_helper<_Ex*>
    {
      static const nested_exception*
      _S_get(const _Ex* __ex)
      { return dynamic_cast<const nested_exception*>(__ex); }
    };

  template<typename _Ex>
    inline const nested_exception*
    __get_nested_exception(const _Ex& __ex)
    { return __get_nested_helper<_Ex>::_S_get(__ex); }

  template<typename _Ex>
    void
    __throw_with_nested(_Ex&&, const nested_exception* = 0)
    __attribute__ ((__noreturn__));

  template<typename _Ex>
    void
    __throw_with_nested(_Ex&&, ...) __attribute__ ((__noreturn__));

  // This function should never be called, but is needed to avoid a warning
  // about ambiguous base classes when instantiating throw_with_nested<_Ex>()
  // with a type that has an accessible nested_exception base.
//   template<typename _Ex>
//     inline void
//     __throw_with_nested(_Ex&& __ex, const nested_exception* = 0)
//     { throw __ex; }

  template<typename _Ex>
    inline void
    __throw_with_nested(_Ex&& __ex, ...)
    { throw _Nested_exception<_Ex>(static_cast<_Ex&&>(__ex)); }
  
  template<typename _Ex>
    void
    throw_with_nested(_Ex __ex) __attribute__ ((__noreturn__));

  /// If @p __ex is derived from nested_exception, @p __ex. 
  /// Else, an implementation-defined object derived from both.
  template<typename _Ex>
    inline void
    throw_with_nested(_Ex __ex)
    {
      if (__get_nested_exception(__ex))
        throw __ex;
      __throw_with_nested(static_cast<_Ex&&>(__ex), &__ex);
    }

  /// If @p __ex is derived from nested_exception, @p __ex.rethrow_nested().
  template<typename _Ex>
    inline void
    rethrow_if_nested(const _Ex& __ex)
    {
      if (const nested_exception* __nested = __get_nested_exception(__ex))
        __nested->rethrow_nested();
    }

  /// Overload, See N2619
  inline void
  rethrow_if_nested(const nested_exception& __ex)
  { __ex.rethrow_nested(); }

  // @} group exceptions
} // namespace std

} // extern "C++"

#endif // __GXX_EXPERIMENTAL_CXX0X__

#pragma GCC visibility pop

#endif


#include <iostream>

#include "lenet5.h"
#include "parameters.h"

void lenet5(
    input_t conv2d_input[N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1],
    result_t layer14_out[N_LAYER_13],
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=conv2d_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=conv2d_input,layer14_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = N_LAYER_13;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<conv2d_weight_t, 150>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 6>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight_default_t, 1200>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 8>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight_default_t, 15360>(w9, "w9.txt");
        nnet::load_weights_from_txt<bias9_t, 120>(b9, "b9.txt");
        nnet::load_weights_from_txt<weight_default_t, 10080>(w11, "w11.txt");
        nnet::load_weights_from_txt<bias11_t, 84>(b11, "b11.txt");
        nnet::load_weights_from_txt<weight_default_t, 840>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 10>(b13, "b13.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer2_t layer2_out[OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::conv_2d_cl<input_t, layer2_t, config2>(conv2d_input, layer2_out, w2, b2); // conv2d

    layer4_t layer4_out[OUT_HEIGHT_4*OUT_WIDTH_4*N_FILT_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::pooling2d_cl<layer2_t, layer4_t, config4>(layer2_out, layer4_out); // max_pooling2d

    layer5_t layer5_out[OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::conv_2d_cl<layer4_t, layer5_t, config5>(layer4_out, layer5_out, w5, b5); // conv2d_1

    layer7_t layer7_out[OUT_HEIGHT_7*OUT_WIDTH_7*N_FILT_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::pooling2d_cl<layer5_t, layer7_t, config7>(layer5_out, layer7_out); // max_pooling2d_1

    layer9_t layer9_out[N_LAYER_9];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::dense<layer7_t, layer9_t, config9>(layer7_out, layer9_out, w9, b9); // dense

    layer11_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense<layer9_t, layer11_t, config11>(layer9_out, layer11_out, w11, b11); // dense_1

    layer13_t layer13_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::dense<layer11_t, layer13_t, config13>(layer11_out, layer13_out, w13, b13); // dense_2

    nnet::linear<layer13_t, result_t, linear_config14>(layer13_out, layer14_out); // dense_2_linear

}
