#pragma once

struct OptionParams {
    double spot;        // Current stock price.
    double strike;      // Option exercise price.
    double rate;        // Annual continuously compounded risk-free rate.
    double volatility;  // Annualized volatility, expressed as a decimal.
    double maturity;    // Time to expiration in years.
};
