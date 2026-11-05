# Regenerate the ComBat-met reference fixtures used by tests/test_glm.py.
#
# There is no Python implementation of ComBat-met anywhere, so the reference
# output is produced once from the author's R package and committed alongside
# the tests. R is deliberately NOT a test dependency: run this script by hand
# only when the fixtures need to change.
#
# Usage (fixtures generated at reference commit ea6763d3550f5f154063d309b3b3eaf8d40010e0):
#     git clone https://github.com/JmWangBio/ComBatMet.git /tmp/ComBatMet
#     git -C /tmp/ComBatMet checkout ea6763d3550f5f154063d309b3b3eaf8d40010e0
#     COMBATMET_SRC=/tmp/ComBatMet \
#     PKG_LIBS="$(R CMD config LAPACK_LIBS) $(R CMD config BLAS_LIBS) $(R CMD config FLIBS)" \
#     Rscript tests/data/generate_met_fixtures.R
#
# The package's C++ sources are compiled in place with Rcpp::sourceCpp rather
# than installed, so only Rcpp is required (the package's `future` /
# `future.apply` imports are used only when ncores > 1).
#
# Only the scenarios below are used as parity fixtures. The reference's
# `covar_mod` path is deliberately excluded: it builds a model matrix per
# covariate column and keeps every intercept, so the within-batch design comes
# out rank-deficient and silently degrades to intercept-only. combatlearn
# builds that design once, without the duplication, so the two cannot agree
# there and the reference is not the better of the two.

src <- Sys.getenv("COMBATMET_SRC", unset = "/tmp/ComBatMet")
if (!dir.exists(src)) stop("set COMBATMET_SRC to a checkout of JmWangBio/ComBatMet")
out <- Sys.getenv("COMBATLEARN_FIXTURES", unset = "tests/data")
if (!dir.exists(out)) stop("run from the repository root, or set COMBATLEARN_FIXTURES")

Rcpp::sourceCpp(file.path(src, "src", "glm_beta.cpp"))
Rcpp::sourceCpp(file.path(src, "src", "compute_apl_beta.cpp"))
Rcpp::sourceCpp(file.path(src, "src", "interpolator.cpp"))
for (f in c("systematicSubset.R", "maximizeInterpolant_beta.R", "mglmLevenbergBeta.R",
            "glmfitBeta.R", "adjustedProfileLikBeta.R", "dispCoxReidBeta.R",
            "estimateGLMCommonDispBeta.R", "dispCoxReidInterpolateTagwiseBeta.R",
            "estimateGLMTagwiseDispBeta.R", "helper_met.r", "utils_parallel.R",
            "ComBat_met.r")) {
  source(file.path(src, "R", f))
}

set.seed(20240722)
G <- 30
N <- 24
batch <- rep(c("A", "B", "C"), each = 8)
group <- rep(c("ctrl", "case"), times = 12)

# Per-feature, per-batch mean on the logit scale and per-batch precision, plus a
# condition effect, so both the mean and the precision carry a batch signal.
mu_b <- plogis(qlogis(runif(G, 0.2, 0.8)) + matrix(rnorm(G * 3, 0, 0.6), G, 3))
phi_b <- exp(matrix(rnorm(G * 3, log(30), 0.4), G, 3))
geff <- matrix(rnorm(G * 2, 0, 0.5), G, 2)
bi <- as.integer(factor(batch))
gi <- as.integer(factor(group))

bv <- matrix(NA_real_, G, N)
for (g in seq_len(G)) {
  for (j in seq_len(N)) {
    m <- plogis(qlogis(mu_b[g, bi[j]]) + geff[g, gi[j]])
    p <- phi_b[g, bi[j]]
    bv[g, j] <- rbeta(1, m * p, (1 - m) * p)
  }
}
bv[bv <= 0] <- 1e-4
bv[bv >= 1] <- 1 - 1e-4

write.csv(bv, file.path(out, "met_input.csv"), row.names = FALSE)
write.csv(data.frame(batch = batch, group = group),
          file.path(out, "met_meta.csv"), row.names = FALSE)

write.csv(ComBat_met(bv, dtype = "b-value", batch = batch, full_mod = FALSE, ncores = 1),
          file.path(out, "met_expected_plain.csv"), row.names = FALSE)
write.csv(ComBat_met(bv, dtype = "b-value", batch = batch, full_mod = FALSE,
                     ref.batch = "B", ncores = 1),
          file.path(out, "met_expected_refbatch.csv"), row.names = FALSE)
write.csv(ComBat_met(bv, dtype = "b-value", batch = batch, group = group,
                     full_mod = TRUE, ncores = 1),
          file.path(out, "met_expected_condition.csv"), row.names = FALSE)

cat("fixtures written to", normalizePath(out), "\n")
