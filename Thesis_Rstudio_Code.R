# ============================================================
# MASTER THESIS — Complete Analysis Pipeline
# Predicting Occupational Stress in ICU Residents
# Data: TILES-2019 (Yau et al., 2022)
# Author: Bram
# ============================================================

# ============================================================
# SECTION 1: PACKAGES
# ============================================================
library(tidyverse)
library(lubridate)
library(brms)
library(tidybayes)
library(loo)
library(RcppCNPy)
library(bayesplot)
library(performance)
library(ggplot2)

# ============================================================
# SECTION 2: PATHS
# ============================================================
base       <- "Data"
audio_path <- "E:/Tiles2019_Data/tiles-phase2-opendataset-audio/raw-features"
fg_path    <- "E:/Tiles2019_Data/tiles-phase2-opendataset-audio/fg-predictions"

# Create output folders
dir.create("models",  showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

# ============================================================
# SECTION 3: LOAD AND CLEAN DATA
# ============================================================

# --- 3.1 EMA data ---
ema_raw <- read_csv(
  file.path(base, "surveys/scored/EMA/daily_ema.csv.gz"),
  show_col_types = FALSE
)

# End-of-day stress (outcome)
ema_eod <- ema_raw %>%
  filter(survey_type == "endofday") %>%
  mutate(date = as.Date(completed_ts)) %>%
  arrange(id, date, completed_ts) %>%
  group_by(id, date) %>%
  slice(1) %>%
  ungroup() %>%
  select(participant_id = id, date, stress_eod = stress)

# Mid-day stress (predictor)
ema_mid <- ema_raw %>%
  filter(survey_type == "midday") %>%
  mutate(date = as.Date(completed_ts)) %>%
  arrange(id, date, completed_ts) %>%
  group_by(id, date) %>%
  slice(1) %>%
  ungroup() %>%
  select(participant_id = id, date, stress_midday = stress)

# --- 3.2 Baseline / SES ---
demographics <- read_csv(
  file.path(base, "surveys/raw/baseline/demographics.csv.gz"),
  show_col_types = FALSE
) %>%
  group_by(participant_id) %>%
  slice(1) %>%
  ungroup() %>%
  rename(training_year = program_year)

baseline_scored <- read_csv(
  file.path(base, "surveys/scored/baseline/baseline.csv.gz"),
  show_col_types = FALSE
) %>%
  group_by(participant_id) %>%
  slice(1) %>%
  ungroup() %>%
  select(participant_id, pss)

ses <- demographics %>%
  left_join(baseline_scored, by = "participant_id") %>%
  mutate(
    program_im = if_else(program == "IM", 1L, 0L),
    program_er = if_else(program == "ER", 1L, 0L)
  )

# --- 3.3 Fitbit ---
fitbit_files <- list.files(
  file.path(base, "fitbit/daily-summary"),
  pattern    = "\\.csv\\.gz$",
  full.names = TRUE
)

fitbit <- map_dfr(fitbit_files, function(f) {
  pid <- str_extract(basename(f), "^[^.]+")
  read_csv(f, show_col_types = FALSE) %>%
    mutate(participant_id = pid)
}) %>%
  select(
    participant_id,
    date             = Timestamp,
    resting_hr       = RestingHeartRate,
    steps            = NumberSteps,
    sleep_minutes    = SleepMinutesAsleep,
    sleep_efficiency = Sleep1Efficiency
  ) %>%
  group_by(participant_id, date) %>%
  slice(1) %>%
  ungroup()

# --- 3.4 Merge into participant-day dataframe ---
df <- ema_eod %>%
  inner_join(ema_mid, by = c("participant_id", "date")) %>%
  left_join(fitbit,   by = c("participant_id", "date")) %>%
  left_join(ses,      by = "participant_id") %>%
  filter(!is.na(stress_eod), !is.na(stress_midday))

# ============================================================
# SECTION 4: CREATE MODEL DATASETS
# ============================================================

# Z-score function
z <- function(x) (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)

df_model <- df %>%
  mutate(
    stress_eod       = factor(stress_eod, levels = 1:7, ordered = TRUE),
    stress_midday_z  = z(stress_midday),
    training_year_z  = z(training_year),
    pss_z            = z(pss),
    resting_hr_z     = z(resting_hr),
    steps_z          = z(steps),
    sleep_minutes_z  = z(sleep_minutes)
  )

# Model A — full EMA sample
df_A <- df_model

# Model B — complete Fitbit days only
df_B <- df_model %>%
  filter(
    !is.na(resting_hr),
    !is.na(steps),
    !is.na(sleep_minutes)
  )

cat("Model A N:", nrow(df_A), "rows |",
    n_distinct(df_A$participant_id), "participants\n")
cat("Model B N:", nrow(df_B), "rows |",
    n_distinct(df_B$participant_id), "participants\n")

# ============================================================
# SECTION 5: DESCRIPTIVE STATISTICS
# ============================================================

# Outcome distribution
cat("=== END-OF-DAY STRESS (Model A) ===\n")
df_A %>%
  mutate(stress_num = as.numeric(stress_eod)) %>%
  summarise(
    mean   = round(mean(stress_num), 2),
    sd     = round(sd(stress_num), 2),
    median = median(stress_num),
    min    = min(stress_num),
    max    = max(stress_num)
  ) %>% print()

df_A %>%
  count(stress_eod) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  print()

# Spearman correlation
cat("\n=== SPEARMAN CORRELATION: midday vs EOD stress ===\n")
df_A %>%
  mutate(stress_num = as.numeric(stress_eod)) %>%
  summarise(
    rho = round(cor(stress_midday, stress_num,
                    method = "spearman",
                    use    = "complete.obs"), 3)
  ) %>% print()

# Fitbit descriptives
cat("\n=== FITBIT VARIABLES (Model B) ===\n")
df_B %>%
  summarise(
    hr_mean    = round(mean(resting_hr), 2),
    hr_sd      = round(sd(resting_hr), 2),
    steps_mean = round(mean(steps), 0),
    steps_sd   = round(sd(steps), 0),
    sleep_mean = round(mean(sleep_minutes), 1),
    sleep_sd   = round(sd(sleep_minutes), 1)
  ) %>% print()

# Training year and programme
cat("\n=== TRAINING YEAR ===\n")
df_A %>%
  group_by(participant_id) %>%
  slice(1) %>%
  ungroup() %>%
  count(training_year) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  print()

cat("\n=== PROGRAMME ===\n")
df_A %>%
  group_by(participant_id) %>%
  slice(1) %>%
  ungroup() %>%
  count(program) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  print()

# PSS
cat("\n=== BASELINE PSS ===\n")
df_A %>%
  group_by(participant_id) %>%
  slice(1) %>%
  ungroup() %>%
  summarise(
    mean = round(mean(pss, na.rm = TRUE), 2),
    sd   = round(sd(pss, na.rm = TRUE), 2),
    min  = min(pss, na.rm = TRUE),
    max  = max(pss, na.rm = TRUE)
  ) %>% print()

# Days per participant
cat("\n=== DAYS PER PARTICIPANT ===\n")
df_A %>%
  count(participant_id) %>%
  summarise(
    mean_days = round(mean(n), 1),
    sd_days   = round(sd(n), 1),
    min_days  = min(n),
    max_days  = max(n)
  ) %>% print()

# Missing data pattern
cat("\n=== MISSING DATA PATTERN ===\n")
df_model %>%
  mutate(
    fitbit_missing = is.na(resting_hr),
    stress_num     = as.numeric(stress_eod)
  ) %>%
  group_by(fitbit_missing) %>%
  summarise(
    mean_stress_eod    = round(mean(stress_num, na.rm = TRUE), 2),
    mean_stress_midday = round(mean(stress_midday, na.rm = TRUE), 2),
    n = n()
  ) %>% print()

# ============================================================
# SECTION 6: AUDIO DATA LOADING
# ============================================================

audio_feature_cols <- c(
  "F0_sma", "F0env_sma", "pcm_RMSenergy_sma", "pcm_zcr_sma",
  "pcm_intensity_sma", "pcm_loudness_sma",
  "pcm_fftMag_fband250-650", "pcm_fftMag_fband1000-4000",
  "pcm_fftMag_spectralRollOff25.0", "pcm_fftMag_spectralRollOff50.0",
  "pcm_fftMag_spectralRollOff75.0", "pcm_fftMag_spectralRollOff90.0",
  "pcm_fftMag_spectralFlux", "pcm_fftMag_spectralCentroid",
  "pcm_fftMag_spectralEntropy", "pcm_fftMag_spectralVariance",
  "pcm_fftMag_spectralSkewness", "pcm_fftMag_spectralKurtosis",
  "pcm_fftMag_spectralSlope", "pcm_fftMag_psySharpness",
  "pcm_fftMag_spectralHarmonicity", "F0final_sma",
  "voicingFinalUnclipped_sma", "jitterLocal_sma",
  "jitterDDP_sma", "shimmerLocal_sma", "logHNR_sma"
)

# EMA windows for audio loading (6AM to 6AM next day)
ema_windows <- df_B %>%
  select(participant_id, date) %>%
  distinct() %>%
  mutate(
    window_start = as.numeric(as.POSIXct(
      paste(date, "06:00:00"), tz = "America/Los_Angeles")),
    window_end = as.numeric(as.POSIXct(
      paste(date + 1, "05:59:59"), tz = "America/Los_Angeles"))
  )

cat("Starting audio loading — this will take 1-3 hours...\n")

audio_daily <- map_dfr(
  unique(ema_windows$participant_id),
  function(pid) {
    
    pid_audio_path <- file.path(audio_path, pid)
    pid_fg_path    <- file.path(fg_path, pid)
    
    if (!dir.exists(pid_audio_path)) {
      cat("No audio folder for:", pid, "\n")
      return(NULL)
    }
    
    windows <- ema_windows %>% filter(participant_id == pid)
    
    all_files <- list.files(pid_audio_path,
                            pattern    = "\\.csv\\.gz$",
                            full.names = TRUE)
    file_ts   <- as.numeric(str_extract(basename(all_files), "^[0-9]+"))
    
    day_results <- map_dfr(seq_len(nrow(windows)), function(i) {
      
      w_start  <- windows$window_start[i]
      w_end    <- windows$window_end[i]
      ema_date <- windows$date[i]
      
      in_window <- all_files[file_ts >= w_start & file_ts <= w_end]
      
      if (length(in_window) == 0) return(NULL)
      
      day_features <- map_dfr(in_window, function(f) {
        tryCatch({
          
          raw <- read_csv(f, show_col_types = FALSE) %>%
            select(any_of(audio_feature_cols))
          
          fg_file <- file.path(
            pid_fg_path,
            str_replace(basename(f), "\\.csv\\.gz$", ".npy")
          )
          
          if (file.exists(fg_file)) {
            fg_probs <- npyLoad(fg_file)
            if (length(fg_probs) == nrow(raw)) {
              raw <- raw[fg_probs > 0.5, ]
            }
          }
          
          if (nrow(raw) == 0) return(NULL)
          
          raw %>% summarise(across(everything(),
                                   ~mean(.x, na.rm = TRUE)))
          
        }, error = function(e) NULL)
      })
      
      if (is.null(day_features) || nrow(day_features) == 0) return(NULL)
      
      day_features %>%
        summarise(across(everything(), ~mean(.x, na.rm = TRUE))) %>%
        mutate(
          participant_id = pid,
          date           = ema_date,
          n_clips        = length(in_window)
        )
    })
    
    cat("Done:", pid, "— days with audio:", nrow(day_results), "\n")
    day_results
  }
)

cat("\nAudio loading complete!\n")
cat("Participant-days with audio:", nrow(audio_daily), "\n")
cat("Participants with audio:",
    n_distinct(audio_daily$participant_id), "\n")

# Audio days per participant
audio_daily %>%
  count(participant_id) %>%
  summarise(
    mean_days = round(mean(n), 1),
    sd_days   = round(sd(n), 1),
    min_days  = min(n),
    max_days  = max(n)
  ) %>% print()

# ============================================================
# SECTION 7: BUILD df_C
# ============================================================

# Fix special character column names
audio_feature_cols_clean <- str_replace_all(
  audio_feature_cols,
  c("-" = "_", "\\." = "_")
)

audio_cols_z <- paste0(audio_feature_cols_clean, "_z")

df_C <- df_B %>%
  inner_join(audio_daily, by = c("participant_id", "date")) %>%
  rename_with(
    ~ str_replace_all(., c("-" = "_", "\\." = "_")),
    starts_with("pcm_fftMag")
  ) %>%
  mutate(across(all_of(audio_feature_cols_clean), z,
                .names = "{.col}_z"))

cat("\nModel C dataset:\n")
cat("Rows:", nrow(df_C), "\n")
cat("Participants:", n_distinct(df_C$participant_id), "\n")

# Stress distribution in Model C subsample
df_C %>%
  mutate(s = as.numeric(stress_eod)) %>%
  summarise(mean = round(mean(s), 2), sd = round(sd(s), 2)) %>%
  print()

# ============================================================
# SECTION 8: PRIORS
# ============================================================

priors_AB <- c(
  prior(normal(0, 1), class = b),
  prior(normal(0, 1), class = Intercept),
  prior(exponential(1), class = sd)
)

priors_C <- c(
  prior(horseshoe(df = 3, par_ratio = 3/27), class = b),
  prior(normal(0, 1), class = Intercept),
  prior(exponential(1), class = sd)
)

# ============================================================
# SECTION 9: FIT MODELS
# ============================================================

# --- Model A ---
model_A <- brm(
  formula = stress_eod ~ stress_midday_z +
    training_year_z + pss_z +
    program_im + program_er +
    (1 | participant_id),
  data    = df_A,
  family  = cumulative("logit"),
  prior   = priors_AB,
  chains  = 4,
  iter    = 4000,
  warmup  = 1000,
  cores   = 4,
  seed    = 42,
  file    = "models/model_A"
)
cat("Model A done\n")
summary(model_A)

# --- Model B ---
model_B <- brm(
  formula = stress_eod ~ stress_midday_z +
    training_year_z + pss_z +
    program_im + program_er +
    resting_hr_z + steps_z + sleep_minutes_z +
    (1 | participant_id),
  data    = df_B,
  family  = cumulative("logit"),
  prior   = priors_AB,
  chains  = 4,
  iter    = 4000,
  warmup  = 1000,
  cores   = 4,
  seed    = 42,
  file    = "models/model_B"
)
cat("Model B done\n")
summary(model_B)

# --- Model C formula ---
model_C_formula <- as.formula(paste(
  "stress_eod ~ stress_midday_z +",
  "training_year_z + pss_z +",
  "program_im + program_er +",
  "resting_hr_z + steps_z + sleep_minutes_z +",
  paste(audio_cols_z, collapse = " + "),
  "+ (1 | participant_id)"
))

# --- Model C ---
model_C_refined <- brm(
  formula = model_C_formula,
  data    = df_C,
  family  = cumulative("logit"),
  prior   = priors_C,
  chains  = 4,
  iter    = 4000,
  warmup  = 1000,
  cores   = 4,
  seed    = 42,
  control = list(adapt_delta = 0.95),
  file    = "models/model_C_refined"
)
cat("Model C done\n")
summary(model_C_refined)

# ============================================================
# SECTION 10: MODEL COMPARISON
# ============================================================

# Refit A and B on df_C subsample for fair LOO comparison
model_A_comp2 <- brm(
  formula = stress_eod ~ stress_midday_z +
    training_year_z + pss_z +
    program_im + program_er +
    (1 | participant_id),
  data    = df_C,
  family  = cumulative("logit"),
  prior   = priors_AB,
  chains  = 4, iter = 4000, warmup = 1000,
  cores   = 4, seed = 42,
  file    = "models/model_A_comp2"
)

model_B_comp2 <- brm(
  formula = stress_eod ~ stress_midday_z +
    training_year_z + pss_z +
    program_im + program_er +
    resting_hr_z + steps_z + sleep_minutes_z +
    (1 | participant_id),
  data    = df_C,
  family  = cumulative("logit"),
  prior   = priors_AB,
  chains  = 4, iter = 4000, warmup = 1000,
  cores   = 4, seed = 42,
  file    = "models/model_B_comp2"
)

# LOO comparison
loo_A2 <- loo(model_A_comp2)
loo_B2 <- loo(model_B_comp2)
loo_C  <- loo(model_C_refined)

cat("\n=== LOO COMPARISON ===\n")
loo_compare(loo_A2, loo_B2, loo_C)

# Bayesian R²
r2_A <- bayes_R2(model_A)
r2_B <- bayes_R2(model_B)
r2_C <- bayes_R2(model_C_refined)

cat("\n=== BAYESIAN R² ===\n")
cat("Model A R²:", round(median(r2_A), 3),
    "95% CI [", round(quantile(r2_A, 0.025), 3), ",",
    round(quantile(r2_A, 0.975), 3), "]\n")
cat("Model B R²:", round(median(r2_B), 3),
    "95% CI [", round(quantile(r2_B, 0.025), 3), ",",
    round(quantile(r2_B, 0.975), 3), "]\n")
cat("Model C R²:", round(median(r2_C), 3),
    "95% CI [", round(quantile(r2_C, 0.025), 3), ",",
    round(quantile(r2_C, 0.975), 3), "]\n")

# ============================================================
# SECTION 11: ICC
# ============================================================

cat("\n=== ICC ===\n")
sd_A <- 0.91
sd_B <- 0.75
sd_C <- 0.44

icc_A <- sd_A^2 / (sd_A^2 + (pi^2 / 3))
icc_B <- sd_B^2 / (sd_B^2 + (pi^2 / 3))
icc_C <- sd_C^2 / (sd_C^2 + (pi^2 / 3))

cat("ICC Model A:", round(icc_A, 3), "\n")
cat("ICC Model B:", round(icc_B, 3), "\n")
cat("ICC Model C:", round(icc_C, 3), "\n")

# Tau with credible intervals
cat("\n=== TAU WITH 95% CREDIBLE INTERVALS ===\n")

posterior_A <- as.data.frame(model_A) %>%
  select(starts_with("sd_participant"))
cat("Model A τ:",
    round(median(posterior_A[[1]]), 2),
    "95% CI [",
    round(quantile(posterior_A[[1]], 0.025), 2), ",",
    round(quantile(posterior_A[[1]], 0.975), 2), "]\n")

posterior_B <- as.data.frame(model_B) %>%
  select(starts_with("sd_participant"))
cat("Model B τ:",
    round(median(posterior_B[[1]]), 2),
    "95% CI [",
    round(quantile(posterior_B[[1]], 0.025), 2), ",",
    round(quantile(posterior_B[[1]], 0.975), 2), "]\n")

posterior_C <- as.data.frame(model_C_refined) %>%
  select(starts_with("sd_participant"))
cat("Model C τ:",
    round(median(posterior_C[[1]]), 2),
    "95% CI [",
    round(quantile(posterior_C[[1]], 0.025), 2), ",",
    round(quantile(posterior_C[[1]], 0.975), 2), "]\n")

# ============================================================
# SECTION 12: COEFFICIENT TABLE
# ============================================================

extract_coefs <- function(model, model_name) {
  model %>%
    gather_draws(`b_.*`, regex = TRUE) %>%
    filter(!grepl("Intercept", .variable)) %>%
    median_qi(.width = 0.95) %>%
    mutate(model = model_name) %>%
    select(model, predictor = .variable,
           median = .value, ci_lower = .lower, ci_upper = .upper)
}

coef_table <- bind_rows(
  extract_coefs(model_A,        "Model A"),
  extract_coefs(model_B,        "Model B"),
  extract_coefs(model_C_refined,"Model C")
) %>%
  mutate(
    predictor = str_remove(predictor, "b_"),
    estimate  = sprintf("%.2f [%.2f, %.2f]",
                        median, ci_lower, ci_upper)
  ) %>%
  select(predictor, model, estimate) %>%
  pivot_wider(names_from  = model,
              values_from = estimate,
              values_fill = "—")

print(coef_table)

# ============================================================
# SECTION 13: POSTERIOR PREDICTIVE CHECKS
# ============================================================

fig_ppc_A <- pp_check(model_A, ndraws = 100, type = "bars") +
  ggtitle("Figure 1. Posterior Predictive Check — Model A") +
  labs(x = "End-of-Day Stress (1–7)", y = "Count") +
  theme_minimal(base_size = 12)

fig_ppc_B <- pp_check(model_B, ndraws = 100, type = "bars") +
  ggtitle("Figure 2. Posterior Predictive Check — Model B") +
  labs(x = "End-of-Day Stress (1–7)", y = "Count") +
  theme_minimal(base_size = 12)

fig_ppc_C <- pp_check(model_C_refined, ndraws = 100, type = "bars") +
  ggtitle("Figure 3. Posterior Predictive Check — Model C") +
  labs(x = "End-of-Day Stress (1–7)", y = "Count") +
  theme_minimal(base_size = 12)

ggsave("figures/Fig1_PPC_ModelA.png", fig_ppc_A,
       width = 6, height = 5, dpi = 300)
ggsave("figures/Fig2_PPC_ModelB.png", fig_ppc_B,
       width = 6, height = 5, dpi = 300)
ggsave("figures/Fig3_PPC_ModelC.png", fig_ppc_C,
       width = 6, height = 5, dpi = 300)

# ============================================================
# SECTION 14: PRESENTATION FIGURES
# ============================================================

# --- Figure: Stress distribution ---
df_A %>%
  count(stress_eod) %>%
  mutate(
    pct        = n / sum(n) * 100,
    stress_num = as.numeric(stress_eod)
  ) %>%
  ggplot(aes(x = stress_eod, y = pct, fill = stress_num)) +
  geom_col(width = 0.7, colour = "white") +
  scale_fill_gradient(low = "#92c5de", high = "#d73027") +
  geom_text(aes(label = paste0(round(pct, 1), "%")),
            vjust = -0.5, size = 4, fontface = "bold") +
  labs(
    title    = "Distribution of End-of-Day Stress",
    subtitle = "642 participant-days, 52 residents",
    x        = "End-of-Day Stress (1–7)",
    y        = "Percentage of Days (%)",
    caption  = "Data: TILES-2019 (Yau et al., 2022)"
  ) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none",
        panel.grid.major.x = element_blank())

ggsave("figures/Fig4_stress_distribution.png",
       width = 7, height = 5, dpi = 300)

# --- Figure: Midday vs EOD stress ---
df_A %>%
  mutate(stress_num = as.numeric(stress_eod)) %>%
  ggplot(aes(x = stress_midday, y = stress_num)) +
  geom_jitter(alpha = 0.25, width = 0.2, height = 0.2,
              colour = "#2166ac", size = 1.5) +
  geom_smooth(method = "lm", colour = "#d73027",
              se = TRUE, linewidth = 1.2) +
  annotate("text", x = 1.5, y = 6.8,
           label = "ρ = .41",
           size = 4.5, fontface = "italic",
           colour = "#d73027") +
  scale_x_continuous(breaks = 1:7) +
  scale_y_continuous(breaks = 1:7) +
  labs(
    title    = "Midday Stress Predicts End-of-Day Stress",
    subtitle = "Spearman's rho = .41",
    x        = "Midday Stress (1–7)",
    y        = "End-of-Day Stress (1–7)",
    caption  = "Each point = one participant-day. Jitter added for visibility."
  ) +
  theme_minimal(base_size = 13)

ggsave("figures/Fig5_midday_vs_eod.png",
       width = 6, height = 6, dpi = 300)

# --- Figure: Sample funnel ---
tibble(
  model = factor(
    c("Model A\nBaseline",
      "Model B\n+ Wearables",
      "Model C\n+ Audio"),
    levels = c("Model A\nBaseline",
               "Model B\n+ Wearables",
               "Model C\n+ Audio")
  ),
  days         = c(642, 513, 296),
  participants = c(52, 47, 36)
) %>%
  pivot_longer(c(days, participants),
               names_to  = "metric",
               values_to = "n") %>%
  mutate(metric = recode(metric,
                         "days"         = "Participant-days",
                         "participants" = "Participants")) %>%
  ggplot(aes(x = model, y = n, fill = model)) +
  geom_col(width = 0.6, colour = "white") +
  geom_text(aes(label = n), vjust = -0.4,
            fontface = "bold", size = 4.5) +
  scale_fill_manual(
    values = c("#4393c3", "#2166ac", "#053061")) +
  facet_wrap(~metric, scales = "free_y") +
  labs(
    title    = "Analytic Sample Across Models",
    subtitle = "Progressive reduction due to missing sensor data",
    x        = NULL,
    y        = "N",
    caption  = "Audio reduction includes opt-outs (21%) and non-compliance (52% rate)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position    = "none",
    panel.grid.major.x = element_blank(),
    strip.text         = element_text(face = "bold", size = 12)
  )

ggsave("figures/Fig6_sample_funnel.png",
       width = 8, height = 5, dpi = 300)

# --- Figure: R² comparison ---
tibble(
  model   = factor(c("Model A", "Model B", "Model C"),
                   levels = c("Model A", "Model B", "Model C")),
  r2      = c(0.281, 0.244, 0.220),
  r2_low  = c(0.044, 0.044, 0.053),
  r2_high = c(0.355, 0.328, 0.338)
) %>%
  ggplot(aes(x = model, y = r2, fill = model)) +
  geom_col(width = 0.5, colour = "white") +
  geom_errorbar(aes(ymin = r2_low, ymax = r2_high),
                width = 0.15, linewidth = 0.8) +
  geom_text(aes(label = round(r2, 3)),
            vjust = -0.8, fontface = "bold", size = 4.5) +
  scale_fill_manual(
    values = c("#4393c3", "#2166ac", "#053061")) +
  scale_y_continuous(
    limits = c(0, 0.45),
    labels = scales::percent_format(accuracy = 1)
  ) +
  labs(
    title    = "Explained Variance Across Models",
    subtitle = "Bayesian R² with 95% credible intervals",
    x        = NULL,
    y        = "Bayesian R²",
    caption  = "R² decreases as more sensor predictors are added."
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position    = "none",
    panel.grid.major.x = element_blank()
  )

ggsave("figures/Fig7_r2_comparison.png",
       width = 6, height = 5, dpi = 300)

# --- Figure: Conditional effects ---
ce <- conditional_effects(
  model_A,
  effects     = "stress_midday_z",
  categorical = TRUE
)

ce_data <- ce$`stress_midday_z:cats__` %>%
  mutate(
    stress_midday_orig = stress_midday_z *
      sd(df_A$stress_midday, na.rm = TRUE) +
      mean(df_A$stress_midday, na.rm = TRUE)
  )

ggplot(ce_data,
       aes(x      = stress_midday_orig,
           y      = estimate__,
           colour = cats__,
           fill   = cats__)) +
  geom_ribbon(aes(ymin = lower__, ymax = upper__),
              alpha = 0.15, colour = NA) +
  geom_line(linewidth = 1.2) +
  scale_colour_brewer(palette = "RdYlGn", direction = -1,
                      name = "End-of-Day\nStress Level") +
  scale_fill_brewer(palette = "RdYlGn", direction = -1,
                    name = "End-of-Day\nStress Level") +
  scale_x_continuous(breaks = 1:7) +
  scale_y_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = c(0, 0.45)
  ) +
  labs(
    title    = "Predicted Stress Probabilities by Midday Stress",
    subtitle = "Model A — cumulative logit predictions",
    x        = "Midday Stress (1–7)",
    y        = "Predicted Probability",
    caption  = "Shaded bands = 95% credible intervals"
  ) +
  theme_minimal(base_size = 13)

ggsave("figures/Fig8_conditional_effects.png",
       width = 8, height = 5, dpi = 300)

cat("\n============================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("Models saved in: models/\n")
cat("Figures saved in: figures/\n")
cat("============================================================\n")