# Usar tidymodels (paqueterC-a)

# Precios de casas saratoga U.S.A. Vamos a hacer particiC3n training test. Datos 2006. 15 variables. 

library(tidyverse)
library(tidymodels)
data("SaratogaHouses", package = "mosaicData")
df <- SaratogaHouses
colnames(df)

# ExploraciC3n y limpieza de los datos respecto a la variable objetivo. 
# Variable objetivo: price
# Esta parte no se cubre porque no es el objetivo de la clase
# DivisiC3n del split de los datos (train/test)
# Semilla
set.seed(123)
# DistribuciC3n de price
df$price %>%  summary()


# Paquete de la botita
split_inicial <- initial_split(data = df, # Datos
                               prop = .8,# ProporciC3n train 
                               strata = price # que los datos se parezcan y la distribucC-C3n de la variable de salida sea parecida
)

print(split_inicial)
data_train <- training(split_inicial)
data_test <- testing(split_inicial)

head(data_train)

data_train$price %>%  summary()
data_test$price %>%  summary()

# resumen de todos los datos
df %>%  summary()

# CategC3ricas no entran en el modelo de regresiC3n, hay que procesar
# varianza cercana a 0 se elimina excepto la flag

# funciC3n receta

# transformaciC3n de datos. Hacer la receta del preprocesado de los datos.
# SeparaciC3n variable objetivo ~ predictores. Puntito es todas las variables
# cocinar ~ ingredientes
transformer <- 
  recipe(formula = price ~ ., data = data_train) %>%  
  step_naomit(all_predictors()) %>% 
  step_nzv(all_date_predictors()) %>% # paso varianza 0
  step_center(all_numeric(), -all_outcomes()) %>%  # centralizar los datos (desviaciC3n estandar)
  # all_outcomes toma todos menos flag
  step_scale(all_numeric(), -all_outcomes()) %>% # escalamiento es con variables numC)ricas
  step_dummy(all_nominal(), -all_outcomes()) # Quita multicolinealidad en R. Las nominales son las categC3ricas. 

# El test nunca se contamina o toca. Fuga de datos data likage
# pre procesado despuC)s de la particiC3n, antes no tocar. 
print(transformer)  

# Se entrena el objeto recipe

transformer_fit <- prep(transformer)

data_train_prep <- bake(transformer_fit, new_data = data_train)
data_test_prep <- bake(transformer_fit, new_data = data_test)
print(data_train_prep)  

##### MODELADO #####

# DefiniciC3n del mC3delo
# Modelo de C!rbol de decisiC3n
# en los modelos en R hay que especificar si es regresiC3n o clasificaciC3n
model_tree <- decision_tree(mode = 'regression') %>%
  set_engine(engine = 'rpart') # AlgorC-tmo para el C!rbol de decisiC3n
print(model_tree)  

# Entrenamiento del modelo
model_tree_fit <- model_tree %>% fit(formula = price ~ .,
                                     data = data_train_prep) # si le puedes meter un dataset normal siempre que sea un C!rbol o un ensamble de C!rboles

print(model_tree_fit) # Esto es un C!rbol de decisiC3n 

# Entrenamiento empleando funciC3n fit_xy
variable_respuesta <- 'price'
predictores <- setdiff(colnames(data_train_prep), variable_respuesta)
print(data_train_prep)
colnames(data_train_prep)
model_tree_fit_2 <- model_tree %>%  fit_xy( x = data_train_prep[,predictores],
                                            y = data_train_prep[[variable_respuesta]]) # se pone [[]] para que lo tome como vector
print(model_tree_fit_2)
print(model_tree_fit)


library(doParallel)
library(tictoc)
library(beepr)

registerDoParallel(cores = detectCores()-1)
set.seed(1234)
cv_folds <- vfold_cv(data_train,
                     v = 5, #partir
                     repeats = 10, #numero de repeticiones de la validacion
                     strata = price)
cv_folds

tic()
validacion_fit <- fit_resamples(object = model_tree, # definicion del modelo 
                                preprocessor = transformer, # se puede poner la receta para transformar los datos
                                resamples = cv_folds, # definicion de la validacion
                                metrics = metric_set(rmse,mae,mape),
                                control = control_resamples(save_pred = TRUE))


grid_fit %>% unnest(.metrics) # nos muestra todos los resultados de los modelos 2x2x3 (2 de V, 2 de grid, 3 metricas)
grid_fit %>% collect_metrics(summarize = TRUE) # Resumen de las metricas (2x2)
grid_fit %>% show_best(metric = "mape", n = 4)

# Pregunta de ex??men

# Cuales son las metricas de clasificacion
# Roc auc, Specificity, Sensitivity
# Cuales son las metricas de regresion

best_model <- select_best(grid_fit, metric = "mae") # Sale profundidad y hojas.Por el hiperparametro que quieras revisar

# Modelo final
# Definicion del modelo
final_model_tree_tune <- finalize_model(x = model_tree_tune,
                                        parameters = best_model
                                        )

final_model_tree_tune

# Entrenar con los mejores hiperparametros
final_model_tree_tune_fit <- final_model_tree_tune %>% fit(formula = price ~ .,
                                                           data = data_train_prep)

final_model_tree_tune_fit

# Presentar resultados en el test
# Calculo de la prediccion - Inferencia - performance en test
predictor <- final_model_tree_tune_fit %>% predict(new_data = data_test_prep,
                                                   type = "numeric", # valor numerico o probabilistico
                                                   )
predictor

# Calculo de los errores de test
# poner el real y poner al lado la prediccion para ver como salio

predicciones <- predictor %>% bind_cols(data_test_prep %>% select(price))
view(predicciones)

rmse(predicciones, truth = price, estimate = .pred)



