# Regenerate the Gaussian ComBat / CovBat reference fixtures used by
# tests/test_combat.py (test_matches_r_reference).
#
# The reference is Andrew Chen's CovBat_Harmonization R code. Its combat() is
# Jean-Philippe Fortin's modification of sva::ComBat, so one checkout covers
# Johnson ComBat (no covariates), Fortin ComBat (with covariates) and CovBat.
# R is deliberately NOT a test dependency: run this script by hand only when the
# fixtures need to change.
#
# Usage (fixtures generated at reference commit cacf370af1ffc87485f50733f797b236c4f251a7):
#     git clone https://github.com/andy1764/CovBat_Harmonization.git /tmp/CovBat_Harmonization
#     git -C /tmp/CovBat_Harmonization checkout cacf370af1ffc87485f50733f797b236c4f251a7
#     COVBAT_SRC=/tmp/CovBat_Harmonization Rscript tests/data/generate_combat_fixtures.R
#
# The sources use four matrixStats functions; when matrixStats is not installed,
# exact base-R equivalents are used instead.
#
# CovBat runs with std.var = FALSE (the PCA of Chen et al. 2022 is on the
# covariance, as in combatlearn), an explicit n.pc (so the two threshold rules
# cannot pick different counts) and score.eb = FALSE, the reference default: a
# plain moment match of the PC scores.

src <- Sys.getenv("COVBAT_SRC", unset = "/tmp/CovBat_Harmonization")
if (!dir.exists(src)) stop("set COVBAT_SRC to a checkout of andy1764/CovBat_Harmonization")
out <- Sys.getenv("COMBATLEARN_FIXTURES", unset = "tests/data")
if (!dir.exists(out)) stop("run from the repository root, or set COMBATLEARN_FIXTURES")

if (requireNamespace("matrixStats", quietly = TRUE)) {
  rowVars <- matrixStats::rowVars
  rowSds <- matrixStats::rowSds
  rowMeans2 <- matrixStats::rowMeans2
  rowSums2 <- matrixStats::rowSums2
} else {
  rowVars <- function(x, na.rm = FALSE, ...) apply(x, 1, var, na.rm = na.rm)
  rowSds <- function(x, na.rm = FALSE, ...) sqrt(rowVars(x, na.rm = na.rm))
  rowMeans2 <- function(x, na.rm = FALSE, ...) rowMeans(x, na.rm = na.rm)
  rowSums2 <- function(x, na.rm = FALSE, ...) rowSums(x, na.rm = na.rm)
}
for (f in c("utils.R", "combat.R", "covbat.R")) source(file.path(src, "R", "R", f))

set.seed(20261103)
G <- 12
per_batch <- 20
batch <- rep(c("A", "B", "C"), each = per_batch)
case_share <- c(A = 0.75, B = 0.5, C = 0.25)
group <- unlist(lapply(names(case_share), function(b) {
  k <- round(case_share[[b]] * per_batch)
  c(rep("case", k), rep("ctrl", per_batch - k))
}))
N <- length(batch)
bi <- as.integer(factor(batch))

# Per-batch location and scale for every feature, a low-rank correlated block
# rescaled per batch (a covariance batch effect), and a group effect on the
# first four features that is unbalanced across batches.
shift <- matrix(rnorm(G * 3, 0, 1.5), G, 3)
scale <- matrix(exp(rnorm(G * 3, 0, 0.3)), G, 3)
loadings <- matrix(rnorm(G * 3), G, 3)
latent_scale <- rbind(c(2, 1, 0.5), c(0.5, 1, 2), c(1, 1, 1))
group_effect <- c(rep(1.5, 4), rep(0, G - 4))

dat <- matrix(NA_real_, G, N)
for (j in seq_len(N)) {
  latent <- loadings %*% (rnorm(3) * latent_scale[bi[j], ])
  dat[, j] <- 5 + shift[, bi[j]] + scale[, bi[j]] * (rnorm(G) + latent) +
    (group[j] == "case") * group_effect
}

write.csv(dat, file.path(out, "combat_input.csv"), row.names = FALSE)
write.csv(data.frame(batch = batch, group = group),
          file.path(out, "combat_meta.csv"), row.names = FALSE)

mod <- model.matrix(~ group)
scenarios <- list(
  johnson_parametric = function() combat(dat, batch, parametric = TRUE)$dat.combat,
  johnson_nonparametric = function() combat(dat, batch, parametric = FALSE)$dat.combat,
  fortin_parametric = function() combat(dat, batch, mod = mod, parametric = TRUE)$dat.combat,
  fortin_nonparametric = function() combat(dat, batch, mod = mod, parametric = FALSE)$dat.combat,
  chen_parametric = function() {
    covbat(dat, batch, mod = mod, n.pc = 3, std.var = FALSE, parametric = TRUE,
           score.eb = FALSE)$dat.covbat
  },
  chen_nonparametric = function() {
    covbat(dat, batch, mod = mod, n.pc = 3, std.var = FALSE, parametric = FALSE,
           score.eb = FALSE)$dat.covbat
  },
  # mean.only applies to the ComBat step only: covbat() harmonizes the mean and
  # the variance of the PC scores regardless.
  fortin_meanonly = function() combat(dat, batch, mod = mod, mean.only = TRUE)$dat.combat,
  chen_meanonly = function() {
    covbat(dat, batch, mod = mod, n.pc = 3, std.var = FALSE, mean.only = TRUE,
           score.eb = FALSE)$dat.covbat
  }
)
for (name in names(scenarios)) {
  write.csv(scenarios[[name]](), file.path(out, paste0("combat_expected_", name, ".csv")),
            row.names = FALSE)
}

cat("fixtures written to", normalizePath(out), "\n")
