library(tidyverse)
library(tidymodels)
library(ISLR) # Datos
library(caTools) #split de los datos
library(ranger) # Paquetería del modelo de Random Forest
#install.packages("ranger")
library(tictoc)
library(beepr)
library(doParallel)

# Mini Exploración de datos
data("Carseats")
# https://rdrr.io/cran/ISLR/man/Carseats.html

df <- Carseats

# Asientos de auto infantiles
# 400 tiendas distintas
# Queremos hacer una clasificación para predecir si una tienda tiene ventas altas
# o ventas bajas
summary(df$Sales)
colnames(df)

# tercer perceptil de ventas
df <- df %>% mutate(sales_high = as.factor(if_else(Sales > 8, "yes", "not"))) %>% 
  select(-Sales)
colnames(df)
summary(df$sales_high)

#### MODELO #######

# Split: División de los datos 
set.seed(123)
split_data <- sample.split(df, SplitRatio = 0.8) # 80% datos de train, 20% test
train <- subset(df, split_data == TRUE) 
test <- subset(df, split_data == FALSE)

###### TUNEADA (OPTIMIZACIÓN) ########
# Optimización de hiperparámetros
# Número de árboles 
# mtry: Variables: Cantidad de variables aleatorias por cada árbol
# max_depth: Máxima profundidad

# no importa como se llaman los hiperparametros
# se nombraran en el ciclo for
grid <- expand_grid('num_trees' = c(50, 100, 500,1000), # número de árboles 
                    'mtry' = c(3,5,7, ncol(df)-1), # hiperparámetros de los features que tenemos N-1 (flag)
                    'max_depth' = c(3,5,7,10)) # máxima profundidad

# Loop para ajustar el modelo a cada combinación de hiperparámetros
# Esto lo hace la paquetería de tiymodels

oob_error = rep(NA, nrow(grid))

tic()
for(i in 1:nrow(grid)){
  
  modelo <- ranger(
    formula = sales_high ~.,
    data = train,
    num.trees = grid$num_trees[i],
    mtry = grid$mtry[i],
    max.depth =  grid$max_depth[i],
    seed = 123
  )
  
  oob_error[i] <- modelo$prediction.error
}
toc()

# Resultados
results <- grid
results$oob_error <- oob_error
results <- results %>% arrange(oob_error)

# Selección de los mejores hiperparámetros
best <- head(results,1)
best

####### GRID PERO BASADA EN VALIDACIÓN CRUZADA CON TIDYMODELS ########

# definición del modelo

model_rf<- rand_forest(
  mode = "classification",
  mtry = tune(),
  trees = tune()# equivalente num.trees
) %>%  set_engine(engine = "ranger",
                  max.depth = tune(),
                  importance = "none",
                  seed = 123 )
  
# preprocesado de datos
transformer <- recipe(formula = sales_high ~.,
                      data = train)

# Validación cruzada 
cv_folds <- vfold_cv(data = train,
                     v = 5,
                     strata = sales_high)
  

# Workflow. Aqui si hay que cuidar nombre de hiperparámetros en la grid
def_workflow <- workflow() %>%  add_recipe(transformer) %>% 
  add_model(model_rf)

grid_hiper <- expand_grid('trees' = c(50, 100, 500,1000), # número de árboles 
                    'mtry' = c(3,5,7, ncol(df)-1), # hiperparámetros de los features que tenemos N-1 (flag)
                    'max.depth' = c(3,5,7,10)) # máxima profundidad
  
  # Optimización de hiperparámetros en el fit
# Número de clusters
cl <- makePSOCKcluster(parallel::detectCores() -1 )
registerDoParallel(cl)  


grid_fit <- tune_grid(
  object = def_workflow,
  resamples = cv_folds, # validación cruzada
  metrics = metric_set(roc_auc,accuracy), 
  grid = grid_hiper
)  

stopCluster(cl)  
  

  
# Mejores hiper
# metricas más importantes para clasificación roc, precision y record. 
# relación de los buenos y los malos
# esto porque los problemas de clasificación son muchas veces
# desbalanceado
# ponemos n=1 porque es el mejor hiperparametro
show_best(grid_fit,metric = "roc_auc", n=1)

best_hiper <- select_best(grid_fit, metric = 'roc_auc')

# Definir MODELO FINAL
final_model <- finalize_workflow(
  x = def_workflow,
  parameters = best_hiper) %>% 
  fit(data = train) %>% extract_fit_parsnip()

# Prediccciones
predicts <- final_model %>% predict(new_data = test) # SI/NO

# Agregarle una columna al lado
predicts <- predicts %>% bind_cols(test %>% select(sales_high))

# efectividad del modelo
accuracy_metric <- accuracy(data = predicts,
                            truth = 'sales_high',
                            estimate = '.pred_class',
                            na_rm = TRUE # no tomar en cuenta NAs
                            )


# Matriz de confusión
# en el caso de no, se equivoca en 9 tiendas. En el caso de si, se equivoca en 10
confu_mat <- conf_mat(
  data = predicts,
  truth = 'sales_high',
  estimate = '.pred_class'
  )

# Predicción de probabilidad
predict_prob <- final_model %>% predict(new_data = test, type = 'prob') # % Prob

# No como 0, si como 1
# se puede mover el punto de corte, por ejemplo decir que arriba de .8 es si
predicts <- predicts %>% cbind(predict_prob)

######## Importancia de los predictores
# Pureza de los nodos. Ginny
# Permutación es más demorado que Ginny
modelo <- rand_forest(mode = 'classification') %>% 
  set_engine(
    engine = 'ranger',
    importance = 'impurity',
    seed = 123
  )

modelo_final <- modelo %>% finalize_model(best_hiper)
modelo_final <- modelo_final %>%  fit(sales_high ~.,
                                      data = train)

# Importancia de los predictores
importance_pred <- modelo_final$fit$variable.importance %>% enframe(name = 'Predictor', value = 'Importance')

# Grafico
importance_pred %>% ggplot(
  aes(x = reorder(Predictor, Importance), y = Importance, fill = Importance)) +
  geom_col() + scale_fill_viridis_c() + coord_flip() +
  theme(legend.position = 'none') + # quitar legend
  labs( x = 'Predictores',
        title = 'Importancia de predictores por pureza de nodos')
  
# Importancia por permutación
modelo_per <- rand_forest(mode = 'classification') %>% 
  set_engine(
    engine = 'ranger',
    importance = 'permutation',
    seed = 123
  )

modelo_final_per <- modelo_per %>% finalize_model(best_hiper)
modelo_final_per <- modelo_final_per %>%  fit(sales_high ~.,
                                      data = train)

# Importancia de los predictores
modelo_final_per$fit$variable.importance

importance_pred_per <- modelo_final_per$fit$variable.importance %>% 
  enframe(name = 'Predictor', value = 'Importance')

# Grafico
# Ejemplo: modelo es cuerpo humano, variable objetivo dolor
# La maestra prefiere esta 
importance_pred_per %>% ggplot(
  aes(x = reorder(Predictor, Importance), y = Importance, fill = Importance)) +
  geom_col() + scale_fill_viridis_c() + coord_flip() +
  theme(legend.position = 'none') + # quitar legend
  labs( x = 'Predictores',
        title = 'Importancia de predictores por permutación')





