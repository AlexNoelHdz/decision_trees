# Usar tidymodels (paquetería)

# Precios de casas saratoga U.S.A. Vamos a hacer partición training test. Datos 2006. 15 variables. 

library(tidyverse)
library(tidymodels)
data("SaratogaHouses", package = "mosaicData")
df <- SaratogaHouses
colnames(df)

# Exploración y limpieza de los datos respecto a la variable objetivo. 
# Variable objetivo: price
# Esta parte no se cubre porque no es el objetivo de la clase
# División del split de los datos (train/test)
# Semilla
set.seed(123)
# Distribución de price
df$price %>%  summary()


# Paquete de la botita
split_inicial <- initial_split(data = df, # Datos
                               prop = .8,# Proporción train 
                               strata = price # que los datos se parezcan y la distribucíón de la variable de salida sea parecida
)

print(split_inicial)
data_train <- training(split_inicial)
data_test <- testing(split_inicial)

head(data_train)

data_train$price %>%  summary()
data_test$price %>%  summary()

# resumen de todos los datos
df %>%  summary()

# Categóricas no entran en el modelo de regresión, hay que procesar
# varianza cercana a 0 se elimina excepto la flag

# función receta

# transformación de datos. Hacer la receta del preprocesado de los datos.
# Separación variable objetivo ~ predictores. Puntito es todas las variables
# cocinar ~ ingredientes
transformer <- recipe(formula = price ~ ., 
                      data = data_train) %>%  
  step_naomit(all_predictors()) %>% 
  step_nzv(all_date_predictors()) %>% # paso varianza 0
  step_center(all_numeric(), -all_outcomes()) %>%  # centralizar los datos (desviación estandar)
  # all_outcomes toma todos menos flag
  step_scale(all_numeric(), -all_outcomes()) %>% # escalamiento es con variables numéricas
  step_dummy(all_nominal(), -all_outcomes()) # Quita multicolinealidad en R. Las nominales son las categóricas. 

# El test nunca se contamina o toca. Fuga de datos data likage
# pre procesado después de la partición, antes no tocar. 
print(transformer)  

# Se entrena el objeto recipe

transformer_fit <- prep(transformer)

data_train_prep <- bake(transformer_fit, new_data = data_train)
data_test_prep <- bake(transformer_fit, new_data = data_test)
print(data_train_prep)  

##### MODELADO #####

# Definición del módelo
# Modelo de árbol de decisión
# en los modelos en R hay que especificar si es regresión o clasificación
model_tree <- decision_tree(mode = 'regression') %>%
  set_engine(engine = 'rpart') # Algorítmo para el árbol de decisión
print(model_tree)  

# Entrenamiento del modelo
model_tree_fit <- model_tree %>% fit(formula = price ~ .,
                                     data = data_train_prep) # si le puedes meter un dataset normal siempre que sea un árbol o un ensamble de árboles

print(model_tree_fit) # Esto es un árbol de decisión 

# Entrenamiento empleando función fit_xy
variable_respuesta <- 'price'
predictores <- setdiff(colnames(data_train_prep), variable_respuesta)
print(data_train_prep)
colnames(data_train_prep)
model_tree_fit_2 <- model_tree %>%  fit_xy( x = data_train_prep[,predictores],
                                            y = data_train_prep[[variable_respuesta]]) # se pone [[]] para que lo tome como vector
print(model_tree_fit_2)
print(model_tree_fit)

set.seed(1234)
cv_folds <- vfold_cv(data_train,
                     v = 5, #partir
                     repeats = 10, #numero de repeticiones de la validacion
                     strata = price)
cv_folds
library(tictoc)
library(beepr)

tic()
validacion_fit <- fit_resamples(object = model_tree, # definición del modelo 
                                preprocessor = transformer, # se puede poner la receta para transformar los datos
                                resamples = cv_folds, # definición de la validación
                                metrics = metric_set(rmse,mae,mape),
                                control = control_resamples(save_pred = TRUE))
toc()
beep(2)
validacion_fit %>% collect_metrics(summarize = TRUE)
# Tarea Por qué es importante que los errores tengan una distribución más o menos normal (no queremos que sea bimodal)




