## ----setup, include=FALSE--------------------------------
options(htmltools.dir.version = FALSE)
knitr::opts_chunk$set(fig.align = "center", dev = "svg")
library(tidyverse)
library(tidymodels)
library(here)
library(knitr)


## ---- echo = FALSE, fig.align="center"-------------------
knitr::include_graphics("https://imgs.xkcd.com/comics/machine_learning.png")


## ---- out.height=400, echo = FALSE-----------------------
 include_graphics(here::here("presentations","imgs","skynet.png"))


## --------------------------------------------------------
caret::getModelInfo() %>% names()


## ---- out.height=400, echo = FALSE-----------------------
 include_graphics(here::here("presentations","imgs","rf.png"))


## ---- echo = FALSE---------------------------------------
include_graphics(here::here("presentations","imgs","flow.png"))


## --------------------------------------------------------
library(AppliedPredictiveModeling)
data(abalone)
abalone <- abalone %>% 
  janitor::clean_names()
glimpse(abalone)


## --------------------------------------------------------

abalone <- abalone %>% 
  mutate(age = rings + 1.5) %>% 
  select(-rings)

abalone %>% 
  ggplot(aes(longest_shell, age, fill = type)) + 
  geom_point(shape  = 21, alpha = 0.75) -> abplot



## ---- echo = FALSE---------------------------------------
abplot


## --------------------------------------------------------
abalone_split <- rsample::initial_split(abalone, prop = 0.75)
abalone_split


## --------------------------------------------------------
abalone_strata_split <-
  rsample::initial_split(abalone,
                         prop = 0.9,
                         strata = "age",
                         breaks = 6)



## ---- echo = FALSE---------------------------------------

include_graphics(here("presentations","imgs","overfit.png"))



## ---- echo = FALSE---------------------------------------

include_graphics(here("presentations","imgs","vfold.png"))




## --------------------------------------------------------

abalone_vfold <- abalone_split %>% 
  rsample::training() %>% 
  rsample::vfold_cv(v = 5, repeats = 2)

abalone_vfold 



## --------------------------------------------------------

abalone_vfold <- abalone_split %>% 
  rsample::training() %>% 
  rsample::vfold_cv(v = 5, repeats = 2, strata = age, breaks = 6)

abalone_vfold %>% 
  mutate(dat = map(splits, analysis))



## ---- echo = FALSE---------------------------------------
include_graphics(here("presentations","imgs","recipes.png"))



## --------------------------------------------------------

analysis_data <- rsample::analysis(abalone_vfold$splits[[1]])

assessment_data <- rsample::assessment(abalone_vfold$splits[[1]])

age_recipe <- recipe(age ~ ., data = analysis_data)

age_recipe



## --------------------------------------------------------

age_recipe <- recipe(age ~ ., data = analysis_data) %>% 
  step_center(all_numeric(), -all_outcomes()) %>% 
  step_scale(all_numeric(),-all_outcomes()) %>% 
  step_dummy(all_nominal())

age_recipe



## --------------------------------------------------------

prepped_abalone <- prep(age_recipe, data = analysis_data)

prepped_abalone


## --------------------------------------------------------

baked_abalone <- bake(prepped_abalone, new_data = analysis_data)

glimpse(baked_abalone)



## --------------------------------------------------------

baked_assessment_abalone <- bake(prepped_abalone, new_data = assessment_data)

glimpse(baked_assessment_abalone)



## ---- echo = TRUE, eval = FALSE--------------------------
## # From randomForest
## rf_1 <-
##   randomForest(x,y,mtry = 12,ntree = 2000,importance = TRUE)
## 
## # From ranger
## rf_2 <- ranger(
##   y ~ .,
##   data = dat,
##   mtry = 12,
##   num.trees = 2000,
##   importance = 'impurity'
## )
## 
## # From sparklyr
## rf_3 <- ml_random_forest(
##   dat,
##   intercept = FALSE,
##   response = "y",
##   features = names(dat)[names(dat) != "y"],
##   col.sample.rate = 12,
##   num.trees = 2000
## )


## ---- echo = FALSE, out.height=200-----------------------
include_graphics(here("presentations","imgs","parsnip.png"))


## --------------------------------------------------------

abalone_xgboost <- parsnip::boost_tree(mode = "regression",mtry = 4, trees = 2000) %>% 
  parsnip::set_engine("xgboost") %>% 
  fit(age ~ ., data = baked_abalone)

abalone_rf <- parsnip::rand_forest(mode = "regression",mtry = ncol(baked_abalone) - 2, trees = 2000, min_n = 2) %>% 
  parsnip::set_engine("ranger") %>% 
  fit(age ~ ., data = baked_abalone)

abalone_lm <- parsnip::linear_reg(mode = "regression") %>% 
  parsnip::set_engine("stan") %>% 
  fit(age ~ ., data = baked_abalone)




## --------------------------------------------------------

analysis_preds <- baked_abalone %>% 
  bind_cols(
    predict(abalone_xgboost, new_data = baked_abalone)
  )

analysis_preds %>% 
  ggplot(aes(age, .pred)) + 
  geom_point() + 
  geom_smooth(method = "lm") + 
  geom_abline(slope = 1, intercept = 0)



## --------------------------------------------------------

assessment_preds <- baked_assessment_abalone %>% 
  bind_cols(
    predict(abalone_lm, new_data = baked_assessment_abalone)
  )

assessment_preds %>% 
  ggplot(aes(age, .pred)) + 
  geom_point() + 
  geom_smooth(method = "lm") + 
  geom_abline(slope = 1, intercept = 0)



## --------------------------------------------------------

assessment_preds %>% 
  mutate(split = "assessment") %>% 
  bind_rows(analysis_preds %>% mutate(split = "analysis")) %>% 
  ggplot(aes(age, .pred, color = split)) + 
  geom_point() + 
  geom_smooth(method = "lm") + 
  geom_abline(slope = 1, intercept = 0)




## --------------------------------------------------------

rf_pred <- predict(abalone_rf, new_data = baked_assessment_abalone) %>% 
  rename(rf_pred = .pred)

lm_pred <- predict(abalone_lm, new_data = baked_assessment_abalone) %>% 
    rename(lm_pred = .pred)

evaluate <- baked_assessment_abalone %>% 
  bind_cols(rf_pred, lm_pred) %>% 
  pivot_longer(cols = contains("_pred"), names_to = "model", values_to = ".pred")

evaluate %>% 
  group_by(model) %>% 
  yardstick::rsq(age, .pred)


fit_foo <- function(mtry, trees, splitrule, min_n, split, recipe, use = "summary"){
  
  analysis_data <- rsample::analysis(split)
  
  assessment_data <- rsample::assessment(split)
  
  prepped_abalone <- prep(recipe, data = analysis_data)
  
  baked_abalone <- bake(prepped_abalone, new_data = analysis_data)
  
  baked_assessment_abalone <-
    bake(prepped_abalone, new_data = assessment_data)
  
  abalone_rf <-
    parsnip::rand_forest(
      mode = "regression",
      mtry = mtry,
      trees = trees,
      min_n = min_n
    ) %>%
    parsnip::set_engine("ranger",
                        splitrule = splitrule) %>%
    fit(age ~ ., data = baked_abalone)
  
  rf_pred <-
    predict(abalone_rf, new_data = baked_assessment_abalone) %>%
    rename(rf_pred_age = .pred) %>%
    bind_cols(assessment_data)
  
  rf_summary <- rf_pred %>%
    summarise(
      rsquared = yardstick::rsq_vec(age, rf_pred_age),
      rmse = yardstick::rmse_vec(age, rf_pred_age)
    )
  
  all_preds <-
    predict(abalone_rf, new_data = baked_assessment_abalone) %>%
    rename(rf_pred_age = .pred) %>%
    bind_cols(assessment_data) %>%
    mutate(split = "assessment") %>%
    bind_rows(
      analysis_data %>%
        bind_cols(predict(abalone_rf, new_data = baked_abalone)) %>%
        rename(rf_pred_age = .pred) %>%
        mutate(split = "analysis")
    )
  
  if (use == "summary") {
    rf_summary
  } else {
    out <- list(
      fit = abalone_rf,
      pred = rf_pred,
      summary = rf_summary,
      all_preds = all_preds
    )
  }
  
}

future::plan(future::multiprocess, workers = 6)

param_grid <- param_grid %>% 
  mutate(rf_fit = future_pmap(list(mtry = mtry, 
                                   trees = trees,
                                   splitrule = splitrule,
                                   min_n = min_n,
                                   split = splits),
                              fit_foo, 
                              recipe = age_recipe,
                              .progress = TRUE))

# try out more things

abalone_vfold$sampid <-
  paste0(abalone_vfold$id, abalone_vfold$id2, sep = '-')

param_grid <- tidyr::expand_grid(
  mtry = seq(1,(ncol(baked_abalone) - 2), by = 1),
  trees = seq(100, 2000, by = 1000),
  splitrule = c("variance", "extratrees"),
  min_n = c(2,10),
  sampid = unique(abalone_vfold$sampid)
) %>%
  left_join(abalone_vfold, by = "sampid")


age_recipe2 <- recipe(age ~ ., data = analysis_data) %>% 
  step_mutate(age_bin = cut(age,6), role = "binner") %>% 
  step_upsample(age_bin,over_ratio = 1) %>% 
  step_corr(all_numeric(),-all_outcomes()) %>% 
  step_BoxCox(all_numeric(), -all_outcomes())
  # step_scale(all_numeric(),-all_outcomes()) %>% 

a = prep(age_recipe2, data = analysis_data, retain = TRUE) %>% 
  juice()

fit_foo <- function(mtry, trees, splitrule, min_n, split, recipe, use = "summary"){
  
  analysis_data <- rsample::analysis(split)
  
  assessment_data <- rsample::assessment(split)
  
  prepped_abalone <- prep(recipe, data = analysis_data)
  
  baked_abalone <- bake(prepped_abalone, new_data = analysis_data) %>% 
    select(-age_bin)
  
  baked_assessment_abalone <-
    bake(prepped_abalone, new_data = assessment_data) %>% 
    select(-age_bin)
  
  abalone_rf <-
    parsnip::rand_forest(
      mode = "regression",
      mtry = min(mtry, ncol(baked_abalone) - 1),
      trees = trees,
      min_n = min_n
    ) %>%
    parsnip::set_engine("ranger",
                        splitrule = splitrule) %>%
    fit(age ~ ., data = baked_abalone)

  rf_pred <-
    predict(abalone_rf, new_data = baked_assessment_abalone) %>%
    rename(rf_pred_age = .pred) %>%
    bind_cols(assessment_data)
  
  rf_summary <- rf_pred %>%
    summarise(
      rsquared = yardstick::rsq_vec(age, rf_pred_age),
      rmse = yardstick::rmse_vec(age, rf_pred_age)
    )
  
  all_preds <-
    predict(abalone_rf, new_data = baked_assessment_abalone) %>%
    rename(rf_pred_age = .pred) %>%
    bind_cols(assessment_data) %>%
    mutate(split = "assessment") %>%
    bind_rows(
      analysis_data %>%
        bind_cols(predict(abalone_rf, new_data = baked_abalone)) %>%
        rename(rf_pred_age = .pred) %>%
        mutate(split = "analysis")
    )
  
  if (use == "summary") {
    rf_summary
  } else {
    out <- list(
      fit = abalone_rf,
      pred = rf_pred,
      summary = rf_summary,
      all_preds = all_preds
    )
  }
  
}

future::plan(future::multiprocess, workers = 6)

param_grid <- param_grid %>%
  mutate(rf_fit = future_pmap(
    list(
      mtry = mtry,
      trees = trees,
      splitrule = splitrule,
      min_n = min_n,
      split = splits
    ),
    (fit_foo),
    recipe = age_recipe2,
    .progress = TRUE
  ))


param_grid %>% 
  unnest(cols = rf_fit) %>% 
  group_by(mtry, trees, min_n, splitrule) %>% 
  summarise(rmse = mean(rmse)) %>% 
  ggplot(aes(mtry, rmse, color = factor(trees))) + 
  geom_line() +
  geom_point() +
  facet_grid(min_n ~ splitrule)



best_params <- param_grid %>% 
  unnest(cols = rf_fit) %>% 
  group_by(mtry, trees, min_n, splitrule) %>% 
  summarise(rmse = mean(rmse)) %>% 
  ungroup() %>% 
  filter(rmse == min(rmse))

best_fit <- fit_foo(mtry = best_params$mtry, trees = best_params$trees,
                    splitrule = best_params$splitrule,
                    min_n = best_params$min_n,
                    recipe = age_recipe2,
                    split = abalone_split, 
                    use = "results")

best_fit$summary$rmse


best_fit$all_preds %>% 
  mutate(model = "Random Forest") %>% 
  rename(pred = rf_pred_age ) %>% 
  ggplot(aes(age, pred, color = model)) + 
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  geom_point() + 
  geom_smooth(method = "lm") +
  facet_wrap(~split)


# try xgboost

abalone_vfold$sampid <-
  paste0(abalone_vfold$id, abalone_vfold$id2, sep = '-')

param_grid <- tidyr::expand_grid(
  mtry = seq(1,(ncol(baked_abalone) - 2), by = 1),
  trees = seq(100, 2000, by = 1000),
  min_n = c(2,10),
  learn_rate = c(0.1, 0.3, 0.6),
  sampid = unique(abalone_vfold$sampid)
) %>%
  left_join(abalone_vfold, by = "sampid")


age_recipe2 <- recipe(age ~ ., data = analysis_data) %>% 
  step_mutate(age_bin = cut(age,6), role = "binner") %>% 
  step_upsample(age_bin,over_ratio = 1) %>% 
  step_center(all_numeric(), -all_outcomes()) %>% 
  step_scale(all_numeric(),-all_outcomes()) %>% 
  step_corr(all_numeric(),-all_outcomes()) %>% 
  step_dummy(all_predictors(),-all_numeric())

a = prep(age_recipe2, data = analysis_data, retain = TRUE) %>% 
  juice()

fit_foo <- function(mtry, trees, learn_rate, min_n, split, recipe, use = "summary"){
  
  analysis_data <- rsample::analysis(split)
  
  assessment_data <- rsample::assessment(split)
  
  prepped_abalone <- prep(recipe, data = analysis_data)
  
  baked_abalone <- bake(prepped_abalone, new_data = analysis_data) %>% 
    select(-age_bin)
  
  baked_assessment_abalone <-
    bake(prepped_abalone, new_data = assessment_data) %>% 
    select(-age_bin)
  
  # abalone_rf <-
  #   parsnip::rand_forest(
  #     mode = "regression",
  #     mtry = min(mtry, ncol(baked_abalone) - 1),
  #     trees = trees,
  #     min_n = min_n
  #   ) %>%
  #   parsnip::set_engine("ranger",
  #                       splitrule = splitrule) %>%
  #   fit(age ~ ., data = baked_abalone)
  
  abalone_rf <-
    parsnip::boost_tree(
      mode = "regression",
      mtry = min(mtry, ncol(baked_abalone) - 1),
      trees = trees,
      min_n = min_n,
      learn_rate = learn_rate
    ) %>%
    parsnip::set_engine("xgboost") %>%
    fit(age ~ ., data = baked_abalone)
  
  rf_pred <-
    predict(abalone_rf, new_data = baked_assessment_abalone) %>%
    rename(rf_pred_age = .pred) %>%
    bind_cols(assessment_data)
  
  rf_summary <- rf_pred %>%
    summarise(
      rsquared = yardstick::rsq_vec(age, rf_pred_age),
      rmse = yardstick::rmse_vec(age, rf_pred_age)
    )
  
  all_preds <-
    predict(abalone_rf, new_data = baked_assessment_abalone) %>%
    rename(rf_pred_age = .pred) %>%
    bind_cols(assessment_data) %>%
    mutate(split = "assessment") %>%
    bind_rows(
      analysis_data %>%
        bind_cols(predict(abalone_rf, new_data = baked_abalone)) %>%
        rename(rf_pred_age = .pred) %>%
        mutate(split = "analysis")
    )
  
  if (use == "summary") {
    rf_summary
  } else {
    out <- list(
      fit = abalone_rf,
      pred = rf_pred,
      summary = rf_summary,
      all_preds = all_preds
    )
  }
  
}

future::plan(future::multiprocess, workers = 6)

param_grid <- param_grid %>%
  mutate(rf_fit = future_pmap(
    list(
      mtry = mtry,
      trees = trees,
      learn_rate = learn_rate,
      min_n = min_n,
      split = splits
    ),
    (fit_foo),
    recipe = age_recipe2,
    .progress = TRUE
  ))


param_grid %>% 
  unnest(cols = rf_fit) %>% 
  group_by(mtry, trees, min_n, learn_rate) %>% 
  summarise(rmse = mean(rmse)) %>% 
  ggplot(aes(mtry, rmse, color = factor(trees))) + 
  geom_line() +
  geom_point() +
  facet_grid(min_n ~ learn_rate)



best_params <- param_grid %>% 
  unnest(cols = rf_fit) %>% 
  group_by(mtry, trees, min_n, learn_rate) %>% 
  summarise(rmse = mean(rmse)) %>% 
  ungroup() %>% 
  filter(rmse == min(rmse))

best_fit <- fit_foo(mtry = best_params$mtry, trees = best_params$trees,
                    learn_rate = best_params$learn_rate,
                    min_n = best_params$min_n,
                    recipe = age_recipe2,
                    split = abalone_split, 
                    use = "results")

best_fit$summary$rmse


best_fit$all_preds %>% 
  mutate(model = "Random Forest") %>% 
  rename(pred = rf_pred_age ) %>% 
  ggplot(aes(age, pred, color = model)) + 
  geom_abline(slope = 1, intercept = 0, linetype = 2) +
  geom_point() + 
  geom_smooth(method = "lm", color = "black", aes(fill = model)) +
  facet_wrap(~split)

