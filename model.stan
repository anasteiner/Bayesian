data {
  int<lower=0> N;                // Number of observations
  int<lower=0> K;                // Number of predictors
  matrix[N, K] X;                // Predictor matrix
  vector[N] y;                   // Response variable (crime counts)
  int<lower=1> num_areas;        // Number of areas
  int<lower=1, upper=num_areas> area[N]; // Area index for each observation
  int<lower=1> num_years;        // Number of years
  int<lower=1, upper=num_years> year[N]; // Year index for each observation
  matrix[N, N] W;                // Spatial weights matrix (adjusted to be NxN)
}

parameters {
  vector[K] beta;                // Regression coefficients
  real alpha;                    // Intercept
  real<lower=0> sigma;           // Residual standard deviation
  real<lower=0> tau_area;        // Standard deviation for area effects
  real<lower=0> tau_year;        // Standard deviation for year effects
  vector[num_areas] u_area;      // Random effects for area
  vector[num_years] u_year;      // Random effects for year
}

model {
  // Priors
  beta ~ normal(0, 5);
  alpha ~ normal(0, 1);
  sigma ~ cauchy(0, 2.5);
  tau_area ~ cauchy(0, 2.5);
  tau_year ~ cauchy(0, 2.5);
  u_area ~ normal(0, tau_area);
  u_year ~ normal(0, tau_year);
  
  // Spatial effects prior (Optional: if using spatial smoothing)
  u_area ~ multi_normal_prec(rep_vector(0, num_areas), W[1:num_areas, 1:num_areas]);
  
  // Likelihood
  y ~ normal(alpha + X * beta + u_area[area] + u_year[year], sigma);
}

generated quantities {
  vector[N] y_pred;
  for (i in 1:N) {
    y_pred[i] = normal_rng(alpha + X[i] * beta + u_area[area[i]] + u_year[year[i]], sigma);
  }
}


