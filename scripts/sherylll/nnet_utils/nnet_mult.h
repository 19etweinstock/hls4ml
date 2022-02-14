#pragma once

namespace nnet {
namespace product{

/* ---
 * 5 different methods to perform the product of input and weight, depending on the
 * types of each. 
 * --- */

template<class x_T, class w_T, class y_T>
class Product{
    public:
    static y_T product(x_T a, w_T w){
        // 'Normal' product
        #pragma HLS INLINE
        return a * w;
    }
    static void limit(unsigned multiplier_limit) {} // Nothing to do here
};
template<class x_T, class w_T, class y_T>
class mult : public Product<x_T, w_T, y_T>{
    public:
    static y_T product(x_T a, w_T w){
        // 'Normal' product
        #pragma HLS INLINE
        return a * w;
    }
    static void limit(unsigned multiplier_limit){
        #pragma HLS INLINE
        #pragma HLS ALLOCATION instances=mul limit=multiplier_limit operation
    }
};

template<class x_T, class w_T, class y_T>
class WAGE_binary : public Product<x_T, w_T, y_T>{
    public:
    static y_T product(x_T a, w_T w){
        // Specialisation for 1-bit data, arbitrary weight
        #pragma HLS INLINE
        return a == 0 ? (w_T) 0 : w;
    }
};
}
}