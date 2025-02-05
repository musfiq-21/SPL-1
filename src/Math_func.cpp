#include "Math_func.h"

namespace neural_autodiff {
    double Math_func::exp(double x) {
        double result = 1.0;
        double term = 1.0;

        for (int i = 1; i <= 10; i++) {
            term *= x/i;
            result += term;
        }

        return result;

    }


    double Math_func::sinh(double x) {
        return (exp(x) - exp(-x))/2;
    }

    double Math_func::cosh(double x) {
        return (exp(x) + exp(-x))/2;
    }

    double Math_func::tanh(double x) {
        return sinh(x)/cosh(x);
    }

};