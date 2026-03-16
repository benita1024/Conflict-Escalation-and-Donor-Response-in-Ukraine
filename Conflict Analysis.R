
# 1. Prep
library(tidyverse)
library(fixest)      
library(modelsummary)
library(ggplot2)


# 2. Data Cleaning
df <- df %>%
  mutate(aid_eur_m = as.numeric(replace_na(aid_eur_m, 0))) %>%
  arrange(donor, aid_type_general, year_month)

# 3. Generate Lags and Leads
df_final <- df %>%
  group_by(donor, aid_type_general) %>%
  mutate(
    lag_1 = lag(infra_spike, 1),
    lag_2 = lag(infra_spike, 2),
    lag_3 = lag(infra_spike, 3),
    lead_1 = lead(infra_spike, 1),
    lead_2 = lead(infra_spike, 2)
  ) %>%
  ungroup() %>%
  replace_na(list(lag_1=0, lag_2=0, lag_3=0, lead_1=0, lead_2=0))

# 4. Distributed Lag Models
# Fixed Effects by Donor | Robust SE (HC1)
reg_hum <- feols(aid_eur_m ~ infra_spike + lag_1 + lag_2 + lag_3 + 
                 battlefield_events + winter + t | donor, 
                 data = subset(df_final, aid_type_general == "Humanitarian"), 
                 vcov = "HC1")

reg_mil <- feols(aid_eur_m ~ infra_spike + lag_1 + lag_2 + lag_3 + 
                 battlefield_events + winter + t | donor, 
                 data = subset(df_final, aid_type_general == "Military"), 
                 vcov = "HC1")

# 5. Generate Table
models <- list("Humanitarian" = reg_hum, "Military" = reg_mil)

modelsummary(models, 
             stars = TRUE,
             coef_map = c("infra_spike" = "Spike (t)",
                         "lag_1" = "Spike (t-1)",
                         "lag_2" = "Spike (t-2)",
                         "lag_3" = "Spike (t-3)",
                         "battlefield_events" = "Battle Intensity",
                         "winter" = "Winter Control",
                         "t" = "Time Trend"),
             gof_map = c("nobs", "r.squared"),
             title = "Table: Regression Results of Donor Responses",
             notes = "Donor Fixed Effects included. Robust SE (HC1) in parentheses.")

# 6. Event Study Visualization
# Creating an event window for Humanitarian aid
event_df <- df_final %>% filter(aid_type_general == "Humanitarian")
event_model <- feols(aid_eur_m ~ lead_2 + lead_1 + infra_spike + 
                     lag_1 + lag_2 + lag_3 | donor, 
                     data = event_df, vcov = "HC1")

# Plot
i_plot <- broom::tidy(event_model, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(time = c(-2, -1, 0, 1, 2, 3))

ggplot(i_plot, aes(x = time, y = estimate)) +
  geom_line(color = "blue") +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.1) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Event Study: Humanitarian Response",
       x = "Months relative to Spike",
       y = "Aid Estimate (M EUR)") +
  theme_bw()
