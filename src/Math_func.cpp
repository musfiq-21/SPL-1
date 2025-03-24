#include "Math_func.h"

namespace neural_autodiff {
    double Math_func::exp(double x) {
        double result = 1.0;
        double term = 1.0;

        for (int i = 1; i <= 21; i++) {
            term *= x/i;
            result += term;
        }

        return result;

    }

    double Math_func::sinh(double x) {
        double exp_x = exp(x);
        double exp_neg_x = exp(-x);

        double result = (exp_x - exp_neg_x)/2;
        return result;
    }

    double Math_func::cosh(double x) {
        double exp_x = exp(x);
        double exp_neg_x = exp(-x);

        double result = (exp_x + exp_neg_x)/2;
        return result;
    }

    double Math_func::tanh(double x) {
        double sinh_x = sinh(x);
        double cosh_x = cosh(x);

        double result = sinh_x / cosh_x;

        return result;
    }

};