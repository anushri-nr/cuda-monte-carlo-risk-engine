#include "cpu_pricer.h"
#include "black_scholes.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        const OptionParams params{100, 100, .05, .2, 1};
        const double analytical = priceEuropeanCallBlackScholes(params);
        require(std::abs(analytical - 10.450583572185565) < 1e-10, "Analytical reference mismatch");
        for (const OptionParams p : {OptionParams{110,100,.05,.2,0},
                                     OptionParams{90,100,.05,.2,0},
                                     OptionParams{100,100,.05,0,1}}) {
            const auto result = priceEuropeanCallCPU(p, 257, 42);
            require(std::abs(result.price - priceEuropeanCallBlackScholes(p)) < 1e-10 &&
                    result.standardError == 0, "Deterministic price mismatch");
        }
        const auto result = priceEuropeanCallCPU(params, 1000000, 42);
        const auto repeat = priceEuropeanCallCPU(params, 1000000, 42);
        require(result.price == repeat.price && result.standardError == repeat.standardError,
                "Reproducibility mismatch");
        require(std::isfinite(result.standardError) && result.standardError > 0 &&
                std::abs(result.price - analytical) < 6 * result.standardError,
                "Statistical reference mismatch");
        for (long long n : {0LL, 1LL}) {
            bool rejected = false;
            try { priceEuropeanCallCPU(params, n, 42); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "Invalid path count accepted");
        }
        auto invalid = params;
        invalid.volatility = std::numeric_limits<double>::quiet_NaN();
        bool rejected = false;
        try { priceEuropeanCallCPU(invalid, 2, 42); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Invalid volatility accepted");
        std::cout << "CPU pricing checks passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
