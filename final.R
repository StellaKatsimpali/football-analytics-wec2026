# ==============================================================================
# WARSAW ECONOMETRIC CHALLENGE 2026
# Project: Decoding the Final 15 Minutes
# Team: The Catharsis Crew (University of Ioannina)
# Script: Master Analytical Pipeline (Feature Eng., ML, SHAP, Econometrics)
# ==============================================================================

# ── 0. LIBRARIES ──────────────────────────────────────────────────────────────
library(tidyverse)
library(caret)
library(xgboost)
library(pROC)
library(shapviz)
library(factoextra)
library(fixest)
library(ggplot2)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

pressure <- read_csv("../data/player_appearance_behaviour_under_pressure.csv", na = c("", "NA", "NULL"))
pass     <- read_csv("../data/player_appearance_pass.csv")
run      <- read_csv("../data/player_appearance_run.csv")
shot     <- read_csv("../data/player_appearance_shot_limited.csv")
quarters <- read_csv("../data/players_quarters_final.csv")


# ── 2. ULTIMATE FEATURE ENGINEERING ───────────────────────────────────────────
cat("\n[1/5] Commencing feature engineering...\n")

# A. Pass Aggregates
pass_agg <- pass %>%
  group_by(player_appearance_id) %>%
  summarise(
    pass_total        = n(),
    pass_accurate     = sum(accurate == TRUE | accurate == "True", na.rm = TRUE),
    pass_accuracy_pct = mean(accurate == TRUE | accurate == "True", na.rm = TRUE),
    .groups = "drop"
  )

# B. Pressure Aggregates
pressure_agg <- pressure %>%
  group_by(player_appearance_id) %>%
  summarise(
    pressure_total         = n(),
    pressure_accurate      = sum(accurate == TRUE | accurate == "True", na.rm = TRUE),
    pressure_turnover      = sum(press_induced_outcome == "turnover", na.rm = TRUE),
    pressure_forward_pass  = sum(press_induced_outcome == "forward_pass", na.rm = TRUE),
    .groups = "drop"
  )

# C. Merge & Create Advanced Ratios (Peak Speed & Intensity)
data_clean <- quarters %>%
  left_join(pass_agg, by = "player_appearance_id") %>%
  left_join(pressure_agg, by = "player_appearance_id") %>%
  #Replacing NA with 0 for the new metrics
  mutate(across(c(starts_with("pass_"), starts_with("pressure_")), ~replace_na(., 0))) %>%
  # Custom Indices (Catharsis Crew Special Features)
  mutate(
    absolute_max_speed = coalesce(cumul_mean_max_speed, 1), # Handle division by zero
    peak_speed_consistency = as.numeric(last15_peak_speed) / absolute_max_speed,
    intensity_index = last15_peak_speed / absolute_max_speed,
    turnover_rate_under_press = ifelse(pressure_total > 0, pressure_turnover / pressure_total, 0),
    scored_after_num = as.numeric(as.character(scored_after))
  ) %>%
  # Clean NA and Inf values across the dataset
  mutate(across(where(is.numeric), ~ ifelse(is.na(.) | is.infinite(.), 0, .)))

# Feature Categorization
col_last15  <- names(data_clean)[str_starts(names(data_clean), "last15_")]
col_cumul   <- names(data_clean)[str_starts(names(data_clean), "cumul_")]
leaky_cols  <- c("player_appearance_id", "player_id", "fixture_id", "date", "scored_after", "scored_after_num", "checkpoint")

# Base Matrix Conversion Helper
to_num_matrix <- function(df) {
  df %>%
    mutate(across(everything(), \(x) {
      if (is.numeric(x))   return(x)
      if (is.logical(x))   return(as.integer(x))
      if (is.Date(x))      return(as.numeric(x))
      return(as.integer(factor(x)))
    })) %>%
    mutate(across(everything(), as.double)) %>%
    as.matrix()
}

# ── 3. ABLATION STUDY (RQ3, RQ4, RQ5) ─────────────────────────────────────────
cat("\n[2/5] Perform ablation study for feature evaluation...\n")

y <- data_clean$scored_after_num
scale_pos <- sum(y == 0) / sum(y == 1)

set.seed(42)
train_idx <- sample(nrow(data_clean), 0.8 * nrow(data_clean))
y_tr <- y[train_idx]
y_te <- y[-train_idx]

# Setup Feature Sets
# 1. Base Model (Keep only Shots & Sprints - RQ3)
shot_sprint_cols <- names(data_clean) %>% keep(\(x) str_detect(x, "shot|sprint|hsr"))
X_base <- to_num_matrix(select(data_clean, all_of(shot_sprint_cols)))

# 2. Extended Model (Shots/Sprints + Passes + Pressure - RQ4)
pass_press_cols <- c(names(select(data_clean, starts_with("pass_"))), 
                     names(select(data_clean, starts_with("pressure_"))))
X_ext <- to_num_matrix(select(data_clean, all_of(c(shot_sprint_cols, pass_press_cols))))

# Train Base Model
dm_base_tr <- xgb.DMatrix(X_base[train_idx, ], label = y_tr)
dm_base_te <- xgb.DMatrix(X_base[-train_idx, ], label = y_te)
model_base <- xgb.train(
  params = list(objective = "binary:logistic", eval_metric = "auc", scale_pos_weight = scale_pos, max_depth = 4, eta = 0.05),
  data = dm_base_tr, nrounds = 100, verbose = 0
)

# Train Extended Model
dm_ext_tr <- xgb.DMatrix(X_ext[train_idx, ], label = y_tr)
dm_ext_te <- xgb.DMatrix(X_ext[-train_idx, ], label = y_te)
model_ext <- xgb.train(
  params = list(objective = "binary:logistic", eval_metric = "auc", scale_pos_weight = scale_pos, max_depth = 4, eta = 0.05),
  data = dm_ext_tr, nrounds = 100, verbose = 0
)

# Extract ROCs and DeLong's Test (The crucial step for RQ4)
roc_base <- roc(y_te, predict(model_base, dm_base_te), quiet = TRUE)
roc_ext  <- roc(y_te, predict(model_ext, dm_ext_te), quiet = TRUE)
delong_test <- roc.test(roc_base, roc_ext, method="delong")

cat("\n══ Comparison Results (Ablation Study) ══\n")
cat("AUC Base Model (Sprints/Shots):", round(auc(roc_base), 4), "\n")
cat("AUC Extended Model (+ Passes/Pressure):", round(auc(roc_ext), 4), "\n")
cat("DeLong's Test p-value:", round(delong_test$p.value, 4), "\n")
if(delong_test$p.value < 0.05) {
  cat("-> CONCLUSION: The addition of passes and pressure SIGNIFICANTLY improves the model!\n")
} else {
  cat("->CONCLUSION: There is an improvement, but the sample size is too small for statistical significance. \n")
}


# ── 4. FINAL EXPLANATORY MODEL & SHAP (RQ1, RQ2) ──────────────────────────────
cat("\n[3/5] Final Model Training & SHAP...\n")

# Select all useful features (excluding IDs)
X_full_cols <- setdiff(names(data_clean), leaky_cols)
X_full <- to_num_matrix(select(data_clean, all_of(X_full_cols)))

dm_full_tr <- xgb.DMatrix(X_full[train_idx, ], label = y_tr)
dm_full_te <- xgb.DMatrix(X_full[-train_idx, ], label = y_te)

final_model <- xgb.train(
  params = list(objective = "binary:logistic", eval_metric = "auc", scale_pos_weight = scale_pos, max_depth = 4, eta = 0.05),
  data = dm_full_tr, nrounds = 150, verbose = 0
)

# Performance Results
preds_final <- predict(final_model, dm_full_te)
roc_final <- roc(y_te, preds_final, quiet = TRUE)
pred_class <- ifelse(preds_final > 0.5, 1, 0)
conf_matrix <- table(Predicted = pred_class, Actual = y_te)
sens <- conf_matrix[2,2] / sum(conf_matrix[,2]) 
spec <- conf_matrix[1,1] / sum(conf_matrix[,1]) 

cat("\n═Evaluate the final model═ ══\n")
cat("AUC:", round(auc(roc_final), 4), "\n")
cat("Balanced Accuracy:", round((sens + spec) / 2, 4), "\n")

# SHAP Analysis
shp <- shapviz(final_model, X_pred = X_full[train_idx, ])


# ── 5. PLAYER ARCHETYPES (CLUSTERING - RQ6) ───────────────────────────────────
cat("\n[4/5] K-Means Clustering (Player Archetypes)...\n")

cluster_data <- data_clean %>%
  filter(peak_speed_consistency < 1.5, intensity_index < 2) %>%
  select(peak_speed_consistency, turnover_rate_under_press, intensity_index) %>%
  drop_na()

vars_scale <- scale(cluster_data)
set.seed(42)
fit_kmeans <- kmeans(vars_scale, centers = 4)


# ── 6. ECONOMETRICS: FIXED EFFECTS (RQ7) ──────────────────────────────────────
cat("\n[5/5] Econometric analysis: Fixed Effects Model...\n")

fe_weight <- sum(data_clean$scored_after_num == 0) / sum(data_clean$scored_after_num == 1)
data_clean$model_weights <- ifelse(data_clean$scored_after_num == 1, fe_weight, 1) 

fe_model <- feglm(scored_after_num ~ peak_speed_consistency + 
                    turnover_rate_under_press + intensity_index | fixture_id + position, 
                  data = data_clean, 
                  family = "binomial", 
                  weights = ~model_weights)

print(summary(fe_model))


# ── 7. EXPORT PLOTS (HIGH-RES) ────────────────────────────────────────────────
cat("\n.[COMPLETION] Save all generated plots to disk..\n")

plot_dir <- "Paper_Plots_Final"
dir.create(plot_dir, showWarnings = FALSE)

# 1. SHAP Beeswarm
p_shap <- sv_importance(shp, kind = "beeswarm", max_display = 15) + theme_minimal()
ggsave(file.path(plot_dir, "1_SHAP_Beeswarm.png"), plot = p_shap, width = 10, height = 7, dpi = 300)

# 2. Clusters
p_clusters <- fviz_cluster(fit_kmeans, data = vars_scale, geom = "point", ellipse.type = "norm", main = "Player Archetypes", ggtheme = theme_minimal())
ggsave(file.path(plot_dir, "2_Player_Clusters.png"), plot = p_clusters, width = 10, height = 7, dpi = 300)

# 3. Final ROC Curve (Base R)
png(file.path(plot_dir, "3_Final_ROC_Curve.png"), width = 2000, height = 2000, res = 300)
plot(roc_final, col="#1c8adb", lwd=3, main=paste("Final Model AUC =", round(auc(roc_final), 3)))
abline(a=0, b=1, lty=2, col="red")
dev.off()

cat("\nSUCCESS! All steps have been completed. The plots are saved in the folder:", plot_dir, "\n")