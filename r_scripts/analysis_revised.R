###############################################################################
# iPVC Manuscript Revision - Comprehensive Analysis Script
# 6 Models: LR, MLP, XGBoost, TabTransformer, TabNet, KAN
#
# This script produces:
#   - Table 2: Discrimination metrics (pre-recalibration)
#   - Table 3: Calibration metrics (pre- and post-recalibration)
#   - Table 4: Overall performance metrics
#   - Table 5: Classification metrics at threshold 0.40
#   - Figure 2: ROC curves (pre-recalibration)
#   - Figure 3: Calibration plots (pre-recalibration)
#   - Figure 4: Decision Curve Analysis (pre-recalibration)
#   - Figure 5: Calibration plots (post-recalibration)
#   - Figure 6: ROC curves (post-recalibration)
#   - Figure 7: DCA (post-recalibration)
#   - Supplementary: Histograms of predicted risk distributions
#   - Supplementary: PVC sensitivity analysis
#   - All metric tables saved as CSV
###############################################################################

# ========================== SETUP ========================== #

# Set working directory to project root
setwd("/Users/apple/Desktop/adem_revizyon")

# Source ONLY the function definitions from perform33.R (first 237 lines)
# Lines 239+ contain demo code with hardcoded paths that would fail
tmp_lines <- readLines("scriptler_data/perform33.R", n = 237)
tmp_file <- tempfile(fileext = ".R")
writeLines(tmp_lines, tmp_file)
source(tmp_file)
unlink(tmp_file)

# Load required packages
library(pacman)
pacman::p_load(
  readr, dplyr, tidyr, ggplot2, gridExtra, grid,
  pROC, PRROC, CalibrationCurves,
  caret, yardstick, rms, ResourceSelection,
  dcurves, viridis, ggbeeswarm
)

# Create output directories if they do not exist
dir.create("revizyon/figures", showWarnings = FALSE, recursive = TRUE)
dir.create("revizyon/outputs", showWarnings = FALSE, recursive = TRUE)

# ========================== COLOR PALETTE (6 models) ========================== #

custom_colors <- c(
  "Logistic_Regression" = "#FF0000",   # Red
  "MLP"                 = "#800080",   # Purple
  "XGBoost"             = "#D2691E",   # Brown/Chocolate (new)
  "TabTransformer"      = "#32CD32",   # Lime Green
  "TabNet"              = "#FF8C42",   # Orange
  "KAN"                 = "#0000FF"    # Blue
)

# Display names for plots (prettier labels)
model_display_names <- c(
  "Logistic_Regression" = "Logistic Regression",
  "MLP"                 = "Multi-Layer Perceptron",
  "XGBoost"             = "XGBoost",
  "TabTransformer"      = "TabTransformer",
  "TabNet"              = "TabNet",
  "KAN"                 = "KAN"
)

# Ordered model list
model_names <- c("Logistic_Regression", "MLP", "XGBoost",
                 "TabTransformer", "TabNet", "KAN")

# Decision threshold
threshold_cut <- 0.40

###############################################################################
#                PART 1: PRE-RECALIBRATION ANALYSIS
###############################################################################

cat("\n========== PART 1: PRE-RECALIBRATION ANALYSIS ==========\n\n")

# ========================== 1.1 DATA LOADING ========================== #

cat("Loading pre-recalibration model predictions...\n")

lr_data     <- read_csv("revizyon/outputs/logistic_regression_results.csv",
                         show_col_types = FALSE)
mlp_data    <- read_csv("revizyon/outputs/mlp_results.csv",
                         show_col_types = FALSE)
xgb_data    <- read_csv("revizyon/outputs/xgboost_results.csv",
                         show_col_types = FALSE)
tt_data     <- read_csv("revizyon/outputs/tabtransformer_results.csv",
                         show_col_types = FALSE)
tabnet_data <- read_csv("revizyon/outputs/tabnet_results.csv",
                         show_col_types = FALSE)
kan_data    <- read_csv("revizyon/outputs/kan_results.csv",
                         show_col_types = FALSE)

# ========================== 1.2 VERIFY TRUE LABELS MATCH ========================== #

cat("Verifying True_Labels consistency across all models...\n")

if (!all(lr_data$True_Labels == mlp_data$True_Labels) ||
    !all(lr_data$True_Labels == xgb_data$True_Labels) ||
    !all(lr_data$True_Labels == tt_data$True_Labels) ||
    !all(lr_data$True_Labels == tabnet_data$True_Labels) ||
    !all(lr_data$True_Labels == kan_data$True_Labels)) {
  stop("ERROR: True_Labels columns do not match across all datasets!")
}
cat("  All True_Labels match. N =", nrow(lr_data), "\n")

# ========================== 1.3 MERGE INTO SINGLE DATAFRAME ========================== #

merged_pre <- data.frame(
  True_Labels         = lr_data$True_Labels,
  Logistic_Regression = lr_data$Predicted_Probabilities,
  MLP                 = mlp_data$Predicted_Probabilities,
  XGBoost             = xgb_data$Predicted_Probabilities,
  TabTransformer      = tt_data$Predicted_Probabilities,
  TabNet              = tabnet_data$Predicted_Probabilities,
  KAN                 = kan_data$Predicted_Probabilities
)

# Clamp exact 0/1 predictions to avoid qlogis() issues
cols_to_clamp <- c("MLP", "XGBoost", "TabTransformer", "TabNet", "KAN")
merged_pre[cols_to_clamp] <- lapply(merged_pre[cols_to_clamp], function(x) {
  ifelse(x == 0, x + 1e-8, ifelse(x == 1, x - 1e-8, x))
})

cat("  Merged dataset dimensions:", dim(merged_pre), "\n\n")

# Save merged dataset
write.csv(merged_pre, "revizyon/outputs/merged_pre_recalibration.csv",
          row.names = FALSE)

# ========================== 1.4 DISCRIMINATION METRICS ========================== #

cat("Computing discrimination metrics (AUROC, AUPRC, AP, pAUROC)...\n")

disc_results_pre <- data.frame()
for (m in model_names) {
  disc_m <- DiscPerfBin(merged_pre$True_Labels, merged_pre[[m]])
  disc_m$Model <- m
  disc_results_pre <- rbind(disc_results_pre, disc_m)
}
disc_results_pre <- disc_results_pre %>% select(Model, everything())
cat("\n--- Discrimination Metrics (Pre-Recalibration) ---\n")
print(disc_results_pre, digits = 4)

write.csv(disc_results_pre,
          "revizyon/outputs/table_discrimination_pre_recalib.csv",
          row.names = FALSE)

# ========================== 1.5 CALIBRATION METRICS ========================== #

cat("\nComputing calibration metrics (O:E, Cal.intercept, Cal.slope, ECI, ICI, ECE)...\n")

cal_results_pre <- data.frame()
for (m in model_names) {
  cal_m <- tryCatch(
    CalPerfBin(merged_pre$True_Labels, merged_pre[[m]], flexcal = "loess"),
    error = function(e) {
      cat("  Warning: CalPerfBin failed for", m, "-", e$message, "\n")
      data.frame(O.E.ratio = NA, Cal..intercept = NA, Cal..slope = NA,
                 ECI = NA, ICI = NA, ECE = NA)
    }
  )
  cal_m$Model <- m
  cal_results_pre <- rbind(cal_results_pre, cal_m)
}
cal_results_pre <- cal_results_pre %>% select(Model, everything())
cat("\n--- Calibration Metrics (Pre-Recalibration) ---\n")
print(cal_results_pre, digits = 4)

write.csv(cal_results_pre,
          "revizyon/outputs/table_calibration_pre_recalib.csv",
          row.names = FALSE)

# ========================== 1.6 OVERALL PERFORMANCE ========================== #

cat("\nComputing overall performance metrics...\n")

ov_results_pre <- data.frame()
for (m in model_names) {
  ov_m <- OvPerfBin(merged_pre$True_Labels, merged_pre[[m]])
  ov_m$Model <- m
  ov_results_pre <- rbind(ov_results_pre, ov_m)
}
ov_results_pre <- ov_results_pre %>% select(Model, everything())
cat("\n--- Overall Performance (Pre-Recalibration) ---\n")
print(ov_results_pre, digits = 4)

write.csv(ov_results_pre,
          "revizyon/outputs/table_overall_pre_recalib.csv",
          row.names = FALSE)

# ========================== 1.7 CLASSIFICATION AT THRESHOLD 0.40 ========================== #

cat("\nComputing classification metrics at threshold =", threshold_cut, "...\n")

class_results_pre <- data.frame()
for (m in model_names) {
  class_m <- ClassPerfBin(merged_pre$True_Labels, merged_pre[[m]],
                           cut = threshold_cut)
  class_m$Model <- m
  class_results_pre <- rbind(class_results_pre, class_m)
}
class_results_pre <- class_results_pre %>% select(Model, everything())
cat("\n--- Classification Metrics at threshold", threshold_cut,
    "(Pre-Recalibration) ---\n")
print(class_results_pre, digits = 4)

write.csv(class_results_pre,
          "revizyon/outputs/table_classification_pre_recalib.csv",
          row.names = FALSE)

# ========================== 1.8 UTILITY METRICS ========================== #

cat("\nComputing utility metrics (Net Benefit, sNB, EC)...\n")

util_results_pre <- data.frame()
for (m in model_names) {
  util_m <- UtilPerfBin(merged_pre$True_Labels, merged_pre[[m]],
                         cut = threshold_cut, costratio = 6)
  util_m$Model <- m
  util_results_pre <- rbind(util_results_pre, util_m)
}
util_results_pre <- util_results_pre %>% select(Model, everything())
cat("\n--- Utility Metrics (Pre-Recalibration) ---\n")
print(util_results_pre, digits = 4)

write.csv(util_results_pre,
          "revizyon/outputs/table_utility_pre_recalib.csv",
          row.names = FALSE)

###############################################################################
#                PART 2: FIGURES (PRE-RECALIBRATION)
###############################################################################

cat("\n========== PART 2: FIGURES (PRE-RECALIBRATION) ==========\n\n")

# ========================== 2.1 FIGURE 2: ROC CURVES ========================== #

cat("Generating Figure 2: ROC Curves (Pre-Recalibration)...\n")

roc_list <- list()
auc_values <- c()

for (m in model_names) {
  roc_obj <- roc(merged_pre$True_Labels, merged_pre[[m]], quiet = TRUE)
  roc_list[[m]] <- data.frame(
    FPR   = 1 - roc_obj$specificities,
    TPR   = roc_obj$sensitivities,
    Model = m
  )
  auc_values[m] <- auc(roc_obj)
}

roc_data <- do.call(rbind, roc_list)

# Sort models by AUC (descending) for legend
sorted_models <- names(sort(auc_values, decreasing = TRUE))
roc_data$Model <- factor(roc_data$Model, levels = sorted_models)

auc_labels <- sprintf("%s (AUC = %.3f)",
                      model_display_names[sorted_models],
                      sort(auc_values, decreasing = TRUE))
names(auc_labels) <- sorted_models

fig2 <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "longdash",
              color = "gray80", linewidth = 0.5, alpha = 0.8) +
  geom_path(linewidth = 1.1, alpha = 0.9) +
  scale_color_manual(values = custom_colors, labels = auc_labels) +
  labs(
    title    = "Model Performance Comparison",
    subtitle = "Receiver Operating Characteristic (ROC) Curves",
    x        = "False Positive Rate (1 - Specificity)",
    y        = "True Positive Rate (Sensitivity)",
    color    = "Models"
  ) +
  theme_minimal(base_family = "Helvetica") +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title    = element_text(size = 18, face = "bold",
                                margin = margin(b = 8), hjust = 0.5),
    plot.subtitle = element_text(size = 13, margin = margin(b = 15),
                                 hjust = 0.5, color = "gray30"),
    axis.title = element_text(size = 12, face = "bold", color = "gray20"),
    axis.text  = element_text(size = 10, color = "gray40"),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.position      = c(0.99, 0.01),
    legend.justification = c(1, 0),
    legend.box.background = element_rect(color = "gray90", fill = "white",
                                          linewidth = 0.3),
    legend.box.margin = margin(6, 6, 6, 6),
    legend.title = element_text(size = 11, face = "bold"),
    legend.text  = element_text(size = 9),
    plot.margin  = margin(t = 15, r = 15, b = 15, l = 15)
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE)

ggsave("revizyon/figures/figure2_roc_pre_recalib.png", fig2,
       width = 10, height = 8, dpi = 300, bg = "white")
ggsave("revizyon/figures/figure2_roc_pre_recalib.pdf", fig2,
       width = 10, height = 8, bg = "white")
cat("  Saved: revizyon/figures/figure2_roc_pre_recalib.png/pdf\n")

# ========================== 2.2 FIGURE 3: CALIBRATION PLOTS (PRE) ========================== #

cat("Generating Figure 3: Calibration Plots (Pre-Recalibration)...\n")

# Helper function to create individual calibration plots
create_calibration_plot <- function(data, title) {
  ggplot(data, aes(x = x, y = y)) +
    geom_abline(aes(intercept = 0, slope = 1, linetype = "Ideal"),
                color = "#404040", linewidth = 0.8) +
    geom_line(aes(color = "Flexible Calibration (Loess)"), linewidth = 1) +
    geom_ribbon(aes(ymin = ymin, ymax = ymax, fill = "95% CI"), alpha = 0.15) +
    scale_color_manual(name = NULL,
                       values = c("Flexible Calibration (Loess)" = "#0066CC")) +
    scale_fill_manual(name = NULL,
                      values = c("95% CI" = "#1F77B4")) +
    scale_linetype_manual(name = NULL,
                          values = c("Ideal" = "dashed")) +
    labs(title = title, x = "Predicted Probability",
         y = "Observed Proportion") +
    theme_minimal() +
    theme(
      legend.position = "none",
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
      axis.text  = element_text(size = 9),
      axis.title = element_text(size = 10)
    )
}

# Generate calibration curve data for each model
cal_curve_data_pre <- list()
cal_objects_pre    <- list()

for (m in model_names) {
  cal_obj <- tryCatch(
    val.prob.ci.2(merged_pre[[m]], merged_pre$True_Labels,
                  logistic.cal = FALSE, smooth = "loess",
                  col.log = "darkblue",
                  allowPerfectPredictions = TRUE),
    error = function(e) {
      cat("  Warning: val.prob.ci.2 failed for", m, "-", e$message, "\n")
      NULL
    }
  )
  if (!is.null(cal_obj)) {
    cal_objects_pre[[m]] <- cal_obj
    cal_curve_data_pre[[m]] <- as.data.frame(
      cal_obj$CalibrationCurves$FlexibleCalibration
    )
  }
}

# Create individual calibration plots
cal_plots_pre <- list()
for (m in model_names) {
  if (!is.null(cal_curve_data_pre[[m]])) {
    cal_plots_pre[[m]] <- create_calibration_plot(
      cal_curve_data_pre[[m]],
      paste("Calibration for", model_display_names[m])
    )
  }
}

# Arrange calibration plots in a grid (3 top, 3 bottom)
if (length(cal_plots_pre) == 6) {
  fig3 <- grid.arrange(
    arrangeGrob(cal_plots_pre[[1]], cal_plots_pre[[2]], cal_plots_pre[[3]],
                ncol = 3, widths = c(1, 1, 1)),
    arrangeGrob(cal_plots_pre[[4]], cal_plots_pre[[5]], cal_plots_pre[[6]],
                ncol = 3, widths = c(1, 1, 1)),
    nrow = 2, heights = c(1, 1)
  )
} else {
  fig3 <- grid.arrange(grobs = cal_plots_pre,
                        ncol = min(3, length(cal_plots_pre)))
}

# Save Figure 3
png("revizyon/figures/figure3_calibration_pre_recalib.png",
    width = 1800, height = 1200, res = 150)
if (length(cal_plots_pre) == 6) {
  grid.arrange(
    arrangeGrob(cal_plots_pre[[1]], cal_plots_pre[[2]], cal_plots_pre[[3]],
                ncol = 3, widths = c(1, 1, 1)),
    arrangeGrob(cal_plots_pre[[4]], cal_plots_pre[[5]], cal_plots_pre[[6]],
                ncol = 3, widths = c(1, 1, 1)),
    nrow = 2, heights = c(1, 1)
  )
} else {
  grid.arrange(grobs = cal_plots_pre, ncol = min(3, length(cal_plots_pre)))
}
dev.off()

pdf("revizyon/figures/figure3_calibration_pre_recalib.pdf",
    width = 18, height = 12)
if (length(cal_plots_pre) == 6) {
  grid.arrange(
    arrangeGrob(cal_plots_pre[[1]], cal_plots_pre[[2]], cal_plots_pre[[3]],
                ncol = 3, widths = c(1, 1, 1)),
    arrangeGrob(cal_plots_pre[[4]], cal_plots_pre[[5]], cal_plots_pre[[6]],
                ncol = 3, widths = c(1, 1, 1)),
    nrow = 2, heights = c(1, 1)
  )
} else {
  grid.arrange(grobs = cal_plots_pre, ncol = min(3, length(cal_plots_pre)))
}
dev.off()

cat("  Saved: revizyon/figures/figure3_calibration_pre_recalib.png/pdf\n")

# ========================== 2.3 FIGURE 4: DCA (PRE-RECALIBRATION) ========================== #

cat("Generating Figure 4: Decision Curve Analysis (Pre-Recalibration)...\n")

fig4 <- dca(
  True_Labels ~ Logistic_Regression + MLP + XGBoost +
    TabTransformer + TabNet + KAN,
  data = merged_pre
) %>%
  plot(smooth = TRUE) +
  scale_color_manual(values = custom_colors,
                     labels = model_display_names) +
  theme_minimal() +
  theme(
    legend.position = "right",
    legend.text     = element_text(size = 10),
    legend.title    = element_text(size = 11, face = "bold"),
    axis.text       = element_text(size = 10),
    axis.title      = element_text(size = 12, face = "bold"),
    plot.background = element_rect(fill = "white", color = NA)
  )

ggsave("revizyon/figures/figure4_dca_pre_recalib.png", fig4,
       width = 10, height = 7, dpi = 300, bg = "white")
ggsave("revizyon/figures/figure4_dca_pre_recalib.pdf", fig4,
       width = 10, height = 7, bg = "white")
cat("  Saved: revizyon/figures/figure4_dca_pre_recalib.png/pdf\n")

###############################################################################
#                PART 3: POST-RECALIBRATION ANALYSIS
###############################################################################

cat("\n========== PART 3: POST-RECALIBRATION ANALYSIS ==========\n\n")

# ========================== 3.1 LOAD RECALIBRATED CSVs ========================== #

cat("Loading post-recalibration model predictions...\n")

lr_recal     <- read_csv("revizyon/outputs/lr_recalibrated_results.csv",
                          show_col_types = FALSE)
mlp_recal    <- read_csv("revizyon/outputs/mlp_recalibrated_results.csv",
                          show_col_types = FALSE)
xgb_recal    <- read_csv("revizyon/outputs/xgboost_recalibrated_results.csv",
                          show_col_types = FALSE)
tt_recal     <- read_csv("revizyon/outputs/tabtransformer_recalibrated_results.csv",
                          show_col_types = FALSE)
tabnet_recal <- read_csv("revizyon/outputs/tabnet_recalibrated_results.csv",
                          show_col_types = FALSE)
kan_recal    <- read_csv("revizyon/outputs/kan_recalibrated_results.csv",
                          show_col_types = FALSE)

# Verify True_Labels consistency
if (!all(lr_recal$True_Labels == mlp_recal$True_Labels) ||
    !all(lr_recal$True_Labels == xgb_recal$True_Labels) ||
    !all(lr_recal$True_Labels == tt_recal$True_Labels) ||
    !all(lr_recal$True_Labels == tabnet_recal$True_Labels) ||
    !all(lr_recal$True_Labels == kan_recal$True_Labels)) {
  stop("ERROR: True_Labels do not match in recalibrated datasets!")
}

# Merge recalibrated predictions
merged_post <- data.frame(
  True_Labels         = lr_recal$True_Labels,
  Logistic_Regression = lr_recal$Predicted_Probabilities,
  MLP                 = mlp_recal$Predicted_Probabilities,
  XGBoost             = xgb_recal$Predicted_Probabilities,
  TabTransformer      = tt_recal$Predicted_Probabilities,
  TabNet              = tabnet_recal$Predicted_Probabilities,
  KAN                 = kan_recal$Predicted_Probabilities
)

# Clamp exact 0/1 predictions
merged_post[cols_to_clamp] <- lapply(merged_post[cols_to_clamp], function(x) {
  ifelse(x == 0, x + 1e-8, ifelse(x == 1, x - 1e-8, x))
})

cat("  Merged recalibrated dataset dimensions:", dim(merged_post), "\n\n")

write.csv(merged_post, "revizyon/outputs/merged_post_recalibration.csv",
          row.names = FALSE)

# ========================== 3.2 POST-RECALIBRATION METRICS ========================== #

cat("Computing post-recalibration discrimination metrics...\n")

disc_results_post <- data.frame()
for (m in model_names) {
  disc_m <- DiscPerfBin(merged_post$True_Labels, merged_post[[m]])
  disc_m$Model <- m
  disc_results_post <- rbind(disc_results_post, disc_m)
}
disc_results_post <- disc_results_post %>% select(Model, everything())
cat("\n--- Discrimination Metrics (Post-Recalibration) ---\n")
print(disc_results_post, digits = 4)
write.csv(disc_results_post,
          "revizyon/outputs/table_discrimination_post_recalib.csv",
          row.names = FALSE)

cat("\nComputing post-recalibration calibration metrics...\n")

cal_results_post <- data.frame()
for (m in model_names) {
  cal_m <- tryCatch(
    CalPerfBin(merged_post$True_Labels, merged_post[[m]], flexcal = "loess"),
    error = function(e) {
      cat("  Warning: CalPerfBin failed for", m, "-", e$message, "\n")
      data.frame(O.E.ratio = NA, Cal..intercept = NA, Cal..slope = NA,
                 ECI = NA, ICI = NA, ECE = NA)
    }
  )
  cal_m$Model <- m
  cal_results_post <- rbind(cal_results_post, cal_m)
}
cal_results_post <- cal_results_post %>% select(Model, everything())
cat("\n--- Calibration Metrics (Post-Recalibration) ---\n")
print(cal_results_post, digits = 4)
write.csv(cal_results_post,
          "revizyon/outputs/table_calibration_post_recalib.csv",
          row.names = FALSE)

cat("\nComputing post-recalibration overall performance...\n")

ov_results_post <- data.frame()
for (m in model_names) {
  ov_m <- OvPerfBin(merged_post$True_Labels, merged_post[[m]])
  ov_m$Model <- m
  ov_results_post <- rbind(ov_results_post, ov_m)
}
ov_results_post <- ov_results_post %>% select(Model, everything())
cat("\n--- Overall Performance (Post-Recalibration) ---\n")
print(ov_results_post, digits = 4)
write.csv(ov_results_post,
          "revizyon/outputs/table_overall_post_recalib.csv",
          row.names = FALSE)

cat("\nComputing post-recalibration classification at threshold =",
    threshold_cut, "...\n")

class_results_post <- data.frame()
for (m in model_names) {
  class_m <- ClassPerfBin(merged_post$True_Labels, merged_post[[m]],
                           cut = threshold_cut)
  class_m$Model <- m
  class_results_post <- rbind(class_results_post, class_m)
}
class_results_post <- class_results_post %>% select(Model, everything())
cat("\n--- Classification Metrics at threshold", threshold_cut,
    "(Post-Recalibration) ---\n")
print(class_results_post, digits = 4)
write.csv(class_results_post,
          "revizyon/outputs/table_classification_post_recalib.csv",
          row.names = FALSE)

cat("\nComputing post-recalibration utility metrics...\n")

util_results_post <- data.frame()
for (m in model_names) {
  util_m <- UtilPerfBin(merged_post$True_Labels, merged_post[[m]],
                         cut = threshold_cut, costratio = 6)
  util_m$Model <- m
  util_results_post <- rbind(util_results_post, util_m)
}
util_results_post <- util_results_post %>% select(Model, everything())
cat("\n--- Utility Metrics (Post-Recalibration) ---\n")
print(util_results_post, digits = 4)
write.csv(util_results_post,
          "revizyon/outputs/table_utility_post_recalib.csv",
          row.names = FALSE)

###############################################################################
#                PART 4: POST-RECALIBRATION FIGURES
###############################################################################

cat("\n========== PART 4: POST-RECALIBRATION FIGURES ==========\n\n")

# ========================== 4.1 FIGURE 5: CALIBRATION PLOTS (POST) ========================== #

cat("Generating Figure 5: Calibration Plots (Post-Recalibration)...\n")

cal_curve_data_post <- list()
cal_objects_post    <- list()

for (m in model_names) {
  cal_obj <- tryCatch(
    val.prob.ci.2(merged_post[[m]], merged_post$True_Labels,
                  logistic.cal = FALSE, smooth = "loess",
                  col.log = "darkblue",
                  allowPerfectPredictions = TRUE),
    error = function(e) {
      cat("  Warning: val.prob.ci.2 failed for", m, "(post-recalib) -",
          e$message, "\n")
      NULL
    }
  )
  if (!is.null(cal_obj)) {
    cal_objects_post[[m]] <- cal_obj
    cal_curve_data_post[[m]] <- as.data.frame(
      cal_obj$CalibrationCurves$FlexibleCalibration
    )
  }
}

cal_plots_post <- list()
for (m in model_names) {
  if (!is.null(cal_curve_data_post[[m]])) {
    cal_plots_post[[m]] <- create_calibration_plot(
      cal_curve_data_post[[m]],
      paste(model_display_names[m], "(Recalibrated)")
    )
  }
}

# Save Figure 5
png("revizyon/figures/figure5_calibration_post_recalib.png",
    width = 1800, height = 1200, res = 150)
if (length(cal_plots_post) == 6) {
  grid.arrange(
    arrangeGrob(cal_plots_post[[1]], cal_plots_post[[2]], cal_plots_post[[3]],
                ncol = 3, widths = c(1, 1, 1)),
    arrangeGrob(cal_plots_post[[4]], cal_plots_post[[5]], cal_plots_post[[6]],
                ncol = 3, widths = c(1, 1, 1)),
    nrow = 2, heights = c(1, 1)
  )
} else {
  grid.arrange(grobs = cal_plots_post, ncol = min(3, length(cal_plots_post)))
}
dev.off()

pdf("revizyon/figures/figure5_calibration_post_recalib.pdf",
    width = 18, height = 12)
if (length(cal_plots_post) == 6) {
  grid.arrange(
    arrangeGrob(cal_plots_post[[1]], cal_plots_post[[2]], cal_plots_post[[3]],
                ncol = 3, widths = c(1, 1, 1)),
    arrangeGrob(cal_plots_post[[4]], cal_plots_post[[5]], cal_plots_post[[6]],
                ncol = 3, widths = c(1, 1, 1)),
    nrow = 2, heights = c(1, 1)
  )
} else {
  grid.arrange(grobs = cal_plots_post, ncol = min(3, length(cal_plots_post)))
}
dev.off()

cat("  Saved: revizyon/figures/figure5_calibration_post_recalib.png/pdf\n")

# ========================== 4.2 FIGURE 6: ROC CURVES (POST) ========================== #

cat("Generating Figure 6: ROC Curves (Post-Recalibration)...\n")

roc_list_post <- list()
auc_values_post <- c()

for (m in model_names) {
  roc_obj <- roc(merged_post$True_Labels, merged_post[[m]], quiet = TRUE)
  roc_list_post[[m]] <- data.frame(
    FPR   = 1 - roc_obj$specificities,
    TPR   = roc_obj$sensitivities,
    Model = m
  )
  auc_values_post[m] <- auc(roc_obj)
}

roc_data_post <- do.call(rbind, roc_list_post)

sorted_models_post <- names(sort(auc_values_post, decreasing = TRUE))
roc_data_post$Model <- factor(roc_data_post$Model, levels = sorted_models_post)

auc_labels_post <- sprintf("%s (AUC = %.3f)",
                           model_display_names[sorted_models_post],
                           sort(auc_values_post, decreasing = TRUE))
names(auc_labels_post) <- sorted_models_post

fig6 <- ggplot(roc_data_post, aes(x = FPR, y = TPR, color = Model)) +
  geom_abline(slope = 1, intercept = 0, linetype = "longdash",
              color = "gray80", linewidth = 0.5, alpha = 0.8) +
  geom_path(linewidth = 1.1, alpha = 0.9) +
  scale_color_manual(values = custom_colors, labels = auc_labels_post) +
  labs(
    title    = "Model Performance Comparison (After Recalibration)",
    subtitle = "Receiver Operating Characteristic (ROC) Curves",
    x        = "False Positive Rate (1 - Specificity)",
    y        = "True Positive Rate (Sensitivity)",
    color    = "Models"
  ) +
  theme_minimal(base_family = "Helvetica") +
  theme(
    plot.background  = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white", color = NA),
    plot.title    = element_text(size = 18, face = "bold",
                                margin = margin(b = 8), hjust = 0.5),
    plot.subtitle = element_text(size = 13, margin = margin(b = 15),
                                 hjust = 0.5, color = "gray30"),
    axis.title = element_text(size = 12, face = "bold", color = "gray20"),
    axis.text  = element_text(size = 10, color = "gray40"),
    panel.grid.major = element_line(color = "gray90", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    legend.position      = c(0.99, 0.01),
    legend.justification = c(1, 0),
    legend.box.background = element_rect(color = "gray90", fill = "white",
                                          linewidth = 0.3),
    legend.box.margin = margin(6, 6, 6, 6),
    legend.title = element_text(size = 11, face = "bold"),
    legend.text  = element_text(size = 9),
    plot.margin  = margin(t = 15, r = 15, b = 15, l = 15)
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE)

ggsave("revizyon/figures/figure6_roc_post_recalib.png", fig6,
       width = 10, height = 8, dpi = 300, bg = "white")
ggsave("revizyon/figures/figure6_roc_post_recalib.pdf", fig6,
       width = 10, height = 8, bg = "white")
cat("  Saved: revizyon/figures/figure6_roc_post_recalib.png/pdf\n")

# ========================== 4.3 FIGURE 7: DCA (POST-RECALIBRATION) ========================== #

cat("Generating Figure 7: DCA (Post-Recalibration)...\n")

fig7 <- dca(
  True_Labels ~ Logistic_Regression + MLP + XGBoost +
    TabTransformer + TabNet + KAN,
  data = merged_post
) %>%
  plot(smooth = TRUE) +
  scale_color_manual(values = custom_colors,
                     labels = model_display_names) +
  theme_minimal() +
  theme(
    legend.position = "right",
    legend.text     = element_text(size = 10),
    legend.title    = element_text(size = 11, face = "bold"),
    axis.text       = element_text(size = 10),
    axis.title      = element_text(size = 12, face = "bold"),
    plot.background = element_rect(fill = "white", color = NA)
  )

ggsave("revizyon/figures/figure7_dca_post_recalib.png", fig7,
       width = 10, height = 7, dpi = 300, bg = "white")
ggsave("revizyon/figures/figure7_dca_post_recalib.pdf", fig7,
       width = 10, height = 7, bg = "white")
cat("  Saved: revizyon/figures/figure7_dca_post_recalib.png/pdf\n")

# ========================== 4.4 COMBINED DCA (PRE + POST SIDE BY SIDE) ========================== #

cat("Generating Combined DCA figure (pre + post side by side)...\n")

png("revizyon/figures/figure_dca_combined.png",
    width = 2000, height = 900, res = 150)
grid.arrange(fig4, fig7, ncol = 2)
dev.off()

pdf("revizyon/figures/figure_dca_combined.pdf", width = 20, height = 9)
grid.arrange(fig4, fig7, ncol = 2)
dev.off()

cat("  Saved: revizyon/figures/figure_dca_combined.png/pdf\n")

###############################################################################
#                PART 5: SUPPLEMENTARY FIGURES
###############################################################################

cat("\n========== PART 5: SUPPLEMENTARY FIGURES ==========\n\n")

# ========================== 5.1 HISTOGRAMS (PRE-RECALIBRATION) ========================== #

cat("Generating Supplementary: Histograms (Pre-Recalibration)...\n")

create_model_histogram <- function(data, model_name, display_name) {
  model_values <- data[[model_name]]
  mean_val   <- mean(model_values, na.rm = TRUE)
  median_val <- median(model_values, na.rm = TRUE)

  ggplot(data.frame(value = model_values), aes(x = value)) +
    geom_histogram(aes(y = after_stat(density)),
                   bins = 30, fill = "skyblue",
                   color = "black", alpha = 0.7) +
    geom_density(color = "blue", linewidth = 1) +
    geom_vline(xintercept = mean_val,
               color = "red", linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = median_val,
               color = "purple", linetype = "dotted", linewidth = 1) +
    annotate("text", x = Inf, y = Inf,
             label = sprintf("Mean: %.3f\nMedian: %.3f", mean_val, median_val),
             hjust = 1.05, vjust = 1.2,
             color = "darkblue", size = 2.8) +
    labs(title = display_name, x = "Predicted Probability", y = "Density") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
      axis.text  = element_text(size = 8),
      axis.title = element_text(size = 9)
    )
}

hist_plots_pre <- lapply(model_names, function(m) {
  create_model_histogram(merged_pre, m, model_display_names[m])
})

# Save histogram grid (3 x 2)
png("revizyon/figures/supp_histograms_pre_recalib.png",
    width = 1500, height = 1000, res = 150)
grid.arrange(
  grobs = hist_plots_pre,
  layout_matrix = rbind(c(1, 2, 3), c(4, 5, 6))
)
dev.off()

pdf("revizyon/figures/supp_histograms_pre_recalib.pdf",
    width = 15, height = 10)
grid.arrange(
  grobs = hist_plots_pre,
  layout_matrix = rbind(c(1, 2, 3), c(4, 5, 6))
)
dev.off()

cat("  Saved: revizyon/figures/supp_histograms_pre_recalib.png/pdf\n")

# ========================== 5.2 HISTOGRAMS (POST-RECALIBRATION) ========================== #

cat("Generating Supplementary: Histograms (Post-Recalibration)...\n")

hist_plots_post <- lapply(model_names, function(m) {
  create_model_histogram(merged_post, m,
                         paste(model_display_names[m], "(Recalibrated)"))
})

png("revizyon/figures/supp_histograms_post_recalib.png",
    width = 1500, height = 1000, res = 150)
grid.arrange(
  grobs = hist_plots_post,
  layout_matrix = rbind(c(1, 2, 3), c(4, 5, 6))
)
dev.off()

pdf("revizyon/figures/supp_histograms_post_recalib.pdf",
    width = 15, height = 10)
grid.arrange(
  grobs = hist_plots_post,
  layout_matrix = rbind(c(1, 2, 3), c(4, 5, 6))
)
dev.off()

cat("  Saved: revizyon/figures/supp_histograms_post_recalib.png/pdf\n")

# ========================== 5.3 DESCRIPTIVE STATISTICS ========================== #

cat("Computing descriptive statistics for predicted probabilities...\n")

# Pre-recalibration
model_stats_pre <- sapply(model_names, function(m) {
  vals <- merged_pre[[m]]
  c(
    Mean   = mean(vals, na.rm = TRUE),
    Median = median(vals, na.rm = TRUE),
    Min    = min(vals, na.rm = TRUE),
    Max    = max(vals, na.rm = TRUE),
    SD     = sd(vals, na.rm = TRUE),
    IQR    = IQR(vals, na.rm = TRUE)
  )
})
write.csv(t(model_stats_pre),
          "revizyon/outputs/descriptive_stats_pre_recalib.csv",
          row.names = TRUE)

# Post-recalibration
model_stats_post <- sapply(model_names, function(m) {
  vals <- merged_post[[m]]
  c(
    Mean   = mean(vals, na.rm = TRUE),
    Median = median(vals, na.rm = TRUE),
    Min    = min(vals, na.rm = TRUE),
    Max    = max(vals, na.rm = TRUE),
    SD     = sd(vals, na.rm = TRUE),
    IQR    = IQR(vals, na.rm = TRUE)
  )
})
write.csv(t(model_stats_post),
          "revizyon/outputs/descriptive_stats_post_recalib.csv",
          row.names = TRUE)

cat("  Saved descriptive statistics CSVs.\n")

###############################################################################
#                PART 6: PVC SENSITIVITY ANALYSIS
###############################################################################

cat("\n========== PART 6: PVC SENSITIVITY ANALYSIS ==========\n\n")

sensitivity_file <- "revizyon/outputs/pvc_sensitivity_results.csv"

if (file.exists(sensitivity_file)) {
  cat("Loading PVC sensitivity results...\n")
  pvc_sens <- read.csv(sensitivity_file)

  cat("  PVC sensitivity data dimensions:", dim(pvc_sens), "\n")
  cat("  Columns:", paste(colnames(pvc_sens), collapse = ", "), "\n\n")

  # Multi-model format: True_Labels, LR_no_PVC, XGBoost_no_PVC, KAN_no_PVC
  sens_model_cols <- setdiff(colnames(pvc_sens), "True_Labels")
  sens_display_names <- c("LR_no_PVC" = "LR (no PVC burden)",
                           "XGBoost_no_PVC" = "XGBoost (no PVC burden)",
                           "KAN_no_PVC" = "KAN (no PVC burden)")

  sens_results <- list()
  for (col in sens_model_cols) {
    cat("\n--- Sensitivity:", sens_display_names[col], "---\n")
    auc_val <- tryCatch({
      roc_obj <- roc(pvc_sens$True_Labels, pvc_sens[[col]], quiet = TRUE)
      auc(roc_obj)
    }, error = function(e) NA)
    cat("  AUROC:", round(as.numeric(auc_val), 4), "\n")

    disc <- tryCatch(DiscPerfBin(pvc_sens$True_Labels, pvc_sens[[col]]),
                     error = function(e) NULL)
    if (!is.null(disc)) {
      cat("  Discrimination metrics computed.\n")
    }

    sens_results[[col]] <- data.frame(
      Model = sens_display_names[col],
      AUROC_no_PVC = round(as.numeric(auc_val), 4)
    )
  }

  # Compare with full-model AUCs
  full_aucs <- c("LR_no_PVC" = as.numeric(auc(roc(merged_pre$True_Labels, merged_pre$Logistic_Regression, quiet = TRUE))),
                  "XGBoost_no_PVC" = as.numeric(auc(roc(merged_pre$True_Labels, merged_pre$XGBoost, quiet = TRUE))),
                  "KAN_no_PVC" = as.numeric(auc(roc(merged_pre$True_Labels, merged_pre$KAN, quiet = TRUE))))

  sens_summary <- do.call(rbind, sens_results)
  sens_summary$AUROC_with_PVC <- round(full_aucs[rownames(sens_summary)], 4)
  sens_summary$AUC_Difference <- sens_summary$AUROC_with_PVC - sens_summary$AUROC_no_PVC
  rownames(sens_summary) <- NULL

  cat("\n--- PVC Sensitivity Summary ---\n")
  print(sens_summary)

  dir.create("revizyon/outputs/sensitivity", showWarnings = FALSE, recursive = TRUE)
  write.csv(sens_summary,
            "revizyon/outputs/sensitivity/pvc_sensitivity_metrics.csv",
            row.names = FALSE)
  cat("  Saved: revizyon/outputs/sensitivity/pvc_sensitivity_metrics.csv\n")
} else {
  cat("  PVC sensitivity file not found at:", sensitivity_file, "\n")
  cat("  Skipping PVC sensitivity analysis.\n")
}

###############################################################################
#                PART 7: BOOTSTRAP CONFIDENCE INTERVALS
###############################################################################

cat("\n========== PART 7: BOOTSTRAP CONFIDENCE INTERVALS ==========\n\n")

run_bootstrap_ci <- function(dataset, dataset_label, nboot = 1000,
                              cut_val = 0.40, costratio_val = 6) {
  cat("Running", nboot, "bootstrap iterations for", dataset_label, "...\n")

  set.seed(42)

  # Get column names from a single run
  ov_example    <- OvPerfBin(dataset$True_Labels,
                              dataset[[model_names[1]]])
  disc_example  <- DiscPerfBin(dataset$True_Labels,
                                dataset[[model_names[1]]])
  cal_example   <- CalPerfBin(dataset$True_Labels,
                               dataset[[model_names[1]]], flexcal = "loess")
  class_example <- ClassPerfBin(dataset$True_Labels,
                                 dataset[[model_names[1]]], cut = cut_val)

  all_colnames <- c(colnames(ov_example), colnames(disc_example),
                    colnames(cal_example), colnames(class_example))

  ci_all_models <- data.frame()

  for (m in model_names) {
    cat("  Bootstrap for model:", m, "\n")

    # Point estimates
    ov_pt    <- OvPerfBin(dataset$True_Labels, dataset[[m]])
    disc_pt  <- DiscPerfBin(dataset$True_Labels, dataset[[m]])
    cal_pt   <- tryCatch(
      CalPerfBin(dataset$True_Labels, dataset[[m]], flexcal = "loess"),
      error = function(e) {
        data.frame(O.E.ratio = NA, Cal..intercept = NA, Cal..slope = NA,
                   ECI = NA, ICI = NA, ECE = NA)
      }
    )
    class_pt <- ClassPerfBin(dataset$True_Labels, dataset[[m]], cut = cut_val)

    point_est <- c(as.numeric(ov_pt), as.numeric(disc_pt),
                   as.numeric(cal_pt), as.numeric(class_pt))

    # Bootstrap matrix
    boot_mat <- matrix(NA, nrow = nboot, ncol = length(point_est))

    for (b in seq_len(nboot)) {
      idx <- sample(nrow(dataset), replace = TRUE)
      bdata <- dataset[idx, ]

      ov_b    <- tryCatch(as.numeric(OvPerfBin(bdata$True_Labels, bdata[[m]])),
                          error = function(e) rep(NA, ncol(ov_pt)))
      disc_b  <- tryCatch(as.numeric(DiscPerfBin(bdata$True_Labels, bdata[[m]])),
                          error = function(e) rep(NA, ncol(disc_pt)))
      cal_b   <- tryCatch(
        as.numeric(CalPerfBin(bdata$True_Labels, bdata[[m]], "loess")),
        error = function(e) rep(NA, 6)
      )
      class_b <- tryCatch(
        as.numeric(ClassPerfBin(bdata$True_Labels, bdata[[m]], cut_val)),
        error = function(e) rep(NA, ncol(class_pt))
      )

      boot_mat[b, ] <- c(ov_b, disc_b, cal_b, class_b)
    }

    # Percentile CI
    lcl <- apply(boot_mat, 2, quantile, probs = 0.025, na.rm = TRUE)
    ucl <- apply(boot_mat, 2, quantile, probs = 0.975, na.rm = TRUE)

    model_ci <- data.frame(
      Model    = m,
      Metric   = all_colnames,
      Estimate = round(point_est, 4),
      LCL      = round(lcl, 4),
      UCL      = round(ucl, 4)
    )
    ci_all_models <- rbind(ci_all_models, model_ci)
  }

  return(ci_all_models)
}

# Pre-recalibration bootstrap
ci_pre <- run_bootstrap_ci(merged_pre, "Pre-Recalibration",
                            nboot = 1000, cut_val = threshold_cut)
write.csv(ci_pre,
          "revizyon/outputs/bootstrap_ci_pre_recalib.csv",
          row.names = FALSE)
cat("  Saved: revizyon/outputs/bootstrap_ci_pre_recalib.csv\n")

# Post-recalibration bootstrap
ci_post <- run_bootstrap_ci(merged_post, "Post-Recalibration",
                             nboot = 1000, cut_val = threshold_cut)
write.csv(ci_post,
          "revizyon/outputs/bootstrap_ci_post_recalib.csv",
          row.names = FALSE)
cat("  Saved: revizyon/outputs/bootstrap_ci_post_recalib.csv\n")

###############################################################################
#                PART 8: COMBINED SUMMARY TABLES
###############################################################################

cat("\n========== PART 8: COMBINED SUMMARY TABLES ==========\n\n")

# ========================== 8.1 AUC COMPARISON TABLE ========================== #

auc_comparison <- data.frame(
  Model                = model_display_names[model_names],
  AUC_Pre_Recalib      = sapply(model_names, function(m) {
    round(auc(roc(merged_pre$True_Labels, merged_pre[[m]], quiet = TRUE)), 4)
  }),
  AUC_Post_Recalib     = sapply(model_names, function(m) {
    round(auc(roc(merged_post$True_Labels, merged_post[[m]], quiet = TRUE)), 4)
  })
)
auc_comparison$AUC_Difference <- auc_comparison$AUC_Post_Recalib -
  auc_comparison$AUC_Pre_Recalib
rownames(auc_comparison) <- NULL

cat("--- AUC Comparison (Pre vs Post Recalibration) ---\n")
print(auc_comparison, digits = 4)

write.csv(auc_comparison,
          "revizyon/outputs/table_auc_comparison.csv",
          row.names = FALSE)

# ========================== 8.2 BRIER SCORE COMPARISON ========================== #

brier_comparison <- data.frame(
  Model                 = model_display_names[model_names],
  Brier_Pre_Recalib     = sapply(model_names, function(m) {
    round(mean((merged_pre$True_Labels - merged_pre[[m]])^2), 4)
  }),
  Brier_Post_Recalib    = sapply(model_names, function(m) {
    round(mean((merged_post$True_Labels - merged_post[[m]])^2), 4)
  })
)
brier_comparison$Brier_Improvement <- brier_comparison$Brier_Pre_Recalib -
  brier_comparison$Brier_Post_Recalib
rownames(brier_comparison) <- NULL

cat("\n--- Brier Score Comparison ---\n")
print(brier_comparison, digits = 4)

write.csv(brier_comparison,
          "revizyon/outputs/table_brier_comparison.csv",
          row.names = FALSE)

# ========================== 8.3 CALIBRATION IMPROVEMENT TABLE ========================== #

cal_improvement <- merge(
  cal_results_pre %>% select(Model, ICI_pre = ICI, ECE_pre = ECE),
  cal_results_post %>% select(Model, ICI_post = ICI, ECE_post = ECE),
  by = "Model"
)
cal_improvement$ICI_Improvement <- cal_improvement$ICI_pre -
  cal_improvement$ICI_post
cal_improvement$ECE_Improvement <- cal_improvement$ECE_pre -
  cal_improvement$ECE_post

cat("\n--- Calibration Improvement Summary ---\n")
print(cal_improvement, digits = 4)

write.csv(cal_improvement,
          "revizyon/outputs/table_calibration_improvement.csv",
          row.names = FALSE)

###############################################################################
#                PART 9: VIOLIN PLOTS (SUPPLEMENTARY)
###############################################################################

cat("\n========== PART 9: VIOLIN PLOTS ==========\n\n")

create_violin_plot <- function(data, model_col, display_name, recalib_label) {
  data$Outcome_Factor <- factor(data$True_Labels,
                                 levels = c(0, 1),
                                 labels = c("Responsive", "Non-Responsive"))

  ggplot(data, aes(x = Outcome_Factor, y = .data[[model_col]])) +
    geom_violin(trim = FALSE, fill = "lightblue", alpha = 0.6) +
    geom_quasirandom(size = 0.8, alpha = 0.5, color = "black") +
    labs(
      x = "",
      y = paste("Estimated risk by", display_name, recalib_label)
    ) +
    theme_minimal() +
    ylim(0, 1) +
    theme(
      axis.title.x = element_text(size = 12, face = "bold"),
      axis.title.y = element_text(size = 10, face = "bold"),
      axis.text    = element_text(size = 10)
    )
}

# Pre-recalibration violins
violin_plots_pre <- lapply(model_names, function(m) {
  create_violin_plot(merged_pre, m, model_display_names[m], "")
})

png("revizyon/figures/supp_violins_pre_recalib.png",
    width = 1800, height = 1200, res = 150)
grid.arrange(
  grobs = violin_plots_pre,
  layout_matrix = rbind(c(1, 2, 3), c(4, 5, 6))
)
dev.off()

# Post-recalibration violins
violin_plots_post <- lapply(model_names, function(m) {
  create_violin_plot(merged_post, m, model_display_names[m], "(Recalibrated)")
})

png("revizyon/figures/supp_violins_post_recalib.png",
    width = 1800, height = 1200, res = 150)
grid.arrange(
  grobs = violin_plots_post,
  layout_matrix = rbind(c(1, 2, 3), c(4, 5, 6))
)
dev.off()

cat("  Saved: revizyon/figures/supp_violins_pre_recalib.png\n")
cat("  Saved: revizyon/figures/supp_violins_post_recalib.png\n")

###############################################################################
#                PART 10: PAIRWISE AUC COMPARISONS (DeLong Test)
###############################################################################

cat("\n========== PART 10: PAIRWISE AUC COMPARISONS ==========\n\n")

# DeLong test for pairwise AUC comparison
pairwise_auc_test <- function(dataset, label) {
  cat("Running DeLong pairwise AUC tests for", label, "...\n")

  pairs <- combn(model_names, 2)
  results <- data.frame()

  for (i in 1:ncol(pairs)) {
    m1 <- pairs[1, i]
    m2 <- pairs[2, i]

    roc1 <- roc(dataset$True_Labels, dataset[[m1]], quiet = TRUE)
    roc2 <- roc(dataset$True_Labels, dataset[[m2]], quiet = TRUE)

    test_result <- tryCatch(
      roc.test(roc1, roc2, method = "delong"),
      error = function(e) {
        cat("  Warning: DeLong test failed for", m1, "vs", m2, "\n")
        list(p.value = NA, statistic = NA)
      }
    )

    results <- rbind(results, data.frame(
      Model_1     = model_display_names[m1],
      Model_2     = model_display_names[m2],
      AUC_1       = round(auc(roc1), 4),
      AUC_2       = round(auc(roc2), 4),
      AUC_Diff    = round(auc(roc1) - auc(roc2), 4),
      DeLong_Z    = round(as.numeric(test_result$statistic), 4),
      P_Value     = round(test_result$p.value, 6)
    ))
  }

  return(results)
}

delong_pre <- pairwise_auc_test(merged_pre, "Pre-Recalibration")
cat("\n--- DeLong Pairwise AUC Tests (Pre-Recalibration) ---\n")
print(delong_pre)
write.csv(delong_pre,
          "revizyon/outputs/table_delong_pre_recalib.csv",
          row.names = FALSE)

delong_post <- pairwise_auc_test(merged_post, "Post-Recalibration")
cat("\n--- DeLong Pairwise AUC Tests (Post-Recalibration) ---\n")
print(delong_post)
write.csv(delong_post,
          "revizyon/outputs/table_delong_post_recalib.csv",
          row.names = FALSE)

###############################################################################
#                FINAL SUMMARY
###############################################################################

cat("\n")
cat("================================================================\n")
cat("                   ANALYSIS COMPLETE                            \n")
cat("================================================================\n\n")

cat("Output files generated:\n\n")

cat("--- Tables (CSV) ---\n")
cat("  revizyon/outputs/merged_pre_recalibration.csv\n")
cat("  revizyon/outputs/merged_post_recalibration.csv\n")
cat("  revizyon/outputs/table_discrimination_pre_recalib.csv\n")
cat("  revizyon/outputs/table_discrimination_post_recalib.csv\n")
cat("  revizyon/outputs/table_calibration_pre_recalib.csv\n")
cat("  revizyon/outputs/table_calibration_post_recalib.csv\n")
cat("  revizyon/outputs/table_overall_pre_recalib.csv\n")
cat("  revizyon/outputs/table_overall_post_recalib.csv\n")
cat("  revizyon/outputs/table_classification_pre_recalib.csv\n")
cat("  revizyon/outputs/table_classification_post_recalib.csv\n")
cat("  revizyon/outputs/table_utility_pre_recalib.csv\n")
cat("  revizyon/outputs/table_utility_post_recalib.csv\n")
cat("  revizyon/outputs/table_auc_comparison.csv\n")
cat("  revizyon/outputs/table_brier_comparison.csv\n")
cat("  revizyon/outputs/table_calibration_improvement.csv\n")
cat("  revizyon/outputs/table_delong_pre_recalib.csv\n")
cat("  revizyon/outputs/table_delong_post_recalib.csv\n")
cat("  revizyon/outputs/bootstrap_ci_pre_recalib.csv\n")
cat("  revizyon/outputs/bootstrap_ci_post_recalib.csv\n")
cat("  revizyon/outputs/descriptive_stats_pre_recalib.csv\n")
cat("  revizyon/outputs/descriptive_stats_post_recalib.csv\n")

cat("\n--- Figures ---\n")
cat("  revizyon/figures/figure2_roc_pre_recalib.png/pdf\n")
cat("  revizyon/figures/figure3_calibration_pre_recalib.png/pdf\n")
cat("  revizyon/figures/figure4_dca_pre_recalib.png/pdf\n")
cat("  revizyon/figures/figure5_calibration_post_recalib.png/pdf\n")
cat("  revizyon/figures/figure6_roc_post_recalib.png/pdf\n")
cat("  revizyon/figures/figure7_dca_post_recalib.png/pdf\n")
cat("  revizyon/figures/figure_dca_combined.png/pdf\n")

cat("\n--- Supplementary ---\n")
cat("  revizyon/figures/supp_histograms_pre_recalib.png/pdf\n")
cat("  revizyon/figures/supp_histograms_post_recalib.png/pdf\n")
cat("  revizyon/figures/supp_violins_pre_recalib.png\n")
cat("  revizyon/figures/supp_violins_post_recalib.png\n")

cat("\n--- Sensitivity (if data available) ---\n")
cat("  revizyon/outputs/sensitivity/pvc_sensitivity_metrics.csv\n")

cat("\n================================================================\n")
cat("  Script finished at:", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "\n")
cat("================================================================\n")
