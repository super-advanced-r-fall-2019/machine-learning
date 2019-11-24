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




