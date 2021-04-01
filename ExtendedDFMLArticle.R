rm(list=ls())
library(tensorflow)
library(keras)
library(forecast)
library(dplyr)
library(abind)
library(vars)
library(DMwR) #unscale function
library(lightgbm)

library(gbcode)
library(ExtendedDFML)

set.seed("171717")

#Make only 1 GPU available to CUDA
Sys.setenv(CUDA_VISIBLE_DEVICES="0") # Comma separated indexes of GPUs to use - GPUs are indexed from 0 to 7 on the workstation
#Set the options to limit the memory allocation on GPUs
tf_gpu_options <- tf$GPUOptions(allow_growth=TRUE, # Don't preallocate memory -> True
                              per_process_gpu_memory_fraction=0.5) # Use up to 0.5*Memory of the GPU
#Create a Tensorflow session with the proper GPU parameters
tf_session <- tf$Session(config=tf$ConfigProto(gpu_options=tf_gpu_options))
#Set the created tensorflow session as Keras backend session
k_set_session(tf_session)

# Get reference to Python garbage collector
py_gc <- reticulate::import("gc")


computeMetrics <- function(cumulative_X_hat,
                           cumulative_X,
                           dimensionality_method,
                           forecasting_method,
                           parameters,
                           dataset_name,
                           components,
                           horizon,
                           columns){
  
  return(
    data.frame(DimensionalityMethod=dimensionality_method,
               ForecastingMethod=forecasting_method,
               Parameters=parameters,
               Dataset=dataset_name,
               Components=components,
               Horizon=horizon,
               Columns=columns,
               WAPE=WAPE(cumulative_X,cumulative_X_hat)$sum,
               MAPE=MAPE(cumulative_X,cumulative_X_hat)$sum,
               SMAPE=SMAPE(cumulative_X,cumulative_X_hat)$sum))
}

source("DirectoriesMapping.R")
source("MultivariateMeasuresArticle.R")

#options(error = quote({dump.frames(); save.image(file = file.path(WORKDIR,paste(paste("dump",dataset_name,sep="_"),"rda",sep="."))); q()}))

#options(error=browser)
profiling <- T
py_gc <- import("gc")


args=(commandArgs(TRUE))
if(length(args) < 1){
 stop("[ERROR] Parameters - <dataset_index>")
}

#datasetIndex <- 1
datasetIndex=as.numeric(args[[1]])

datasets.list <- list.files(path=".",recursive = TRUE,pattern = "*.csv|*.ssv|*.txt")
datasets.list <- datasets.list[grepl("data",datasets.list)]
if (datasetIndex %in% c(1,3)){ #Datasets in TXT format
  dataset_name <- unlist(strsplit(datasets.list[[datasetIndex]],split="[.]"))[1]
  dataset_name <- unlist(strsplit(dataset_name,split="/"))[4]
  
  X <- read.csv(file.path(datasets.list[[datasetIndex]]),header = F)
} else {
  dataset_name <- unlist(strsplit(datasets.list[[datasetIndex]],split="[.]"))[1]
  dataset_name <- unlist(strsplit(dataset_name,split="/"))[1]
  
  X <- read.csv(file.path(datasets.list[[datasetIndex]]),stringsAsFactors = F)
}

# Certain datasets have timestamps, others don't
if(datasetIndex %in% c(2)){
  X_timestamps <- as.POSIXct(X[,1])
  X <- as.matrix(X[,-1])
  # Mobility
  if(datasetIndex == 2){
    X[is.na(X)] <- 0
  }
} else{
  X <- as.matrix(X)
}

horizons <- c(4,6,12,24)
components <- c(3)
n_series <- ncol(X)
epochs <- 50
perform_global_diff <- T
perform_global_scale <- T
perform_local_scale <- F

# Mobility
if(datasetIndex == 2){
  window_size <- 400
}
#Electricity and Traffic
if(datasetIndex %in% c(1,3)){
  window_size <- 2000
}

cat("[INFO] - Performing first order difference and normalization\n")
if(perform_global_diff){
  X <- apply(X,2,diff)
}
if(perform_global_scale){
  X <- scale(X)
}

cat("[INFO] - Running sanity check on the presence of costant values\n")

# Sanity check on constant values
X_const_filter <- apply(apply(X,2,diff) == 0,2,sum)
X_const_filter_RLE <- apply(X,2,function(X_i){rle(X_i)}) # Apply RLE on all the TS
X_const_filter_RLE <- sapply(X_const_filter_RLE,
                             function(X_i_RLE){
                               sum(X_i_RLE$lengths[X_i_RLE$values == 0] > window_size)
                               }) # Check the sequences of O longer than window_size

idx_0_constant_TS <- which(X_const_filter_RLE > 0)
if(length(idx_0_constant_TS) > 0){
  cat("[INFO] - Removing columns with 0-constant values:",idx_0_constant_TS,"\n")
  X <- X[,-c(idx_0_constant_TS)]
  n_series <- ncol(X)
}

dimensionality_methods <- c("PCA","lstm","gru","base","deep")
multistepAhead_methods <- MULTISTEPAHEAD_METHODS

ss_results.df <- data.frame(DimensionalityMethod=character(),
                            ForecastingMethod=character(),
                            Parameters=numeric(),
                            Dataset=character(),
                            Components=numeric(),
                            Horizon=numeric(),
                            Columns=numeric(),
                            Time=numeric(),
                            MSE=numeric(),
                            NNMSE=numeric(),
                            Samples=numeric(),
                            stringsAsFactors = FALSE)

article_results.df <- data.frame(DimensionalityMethod=character(),
                                 ForecastingMethod=character(),
                                 Parameters=numeric(),
                                 Dataset=character(),
                                 Components=numeric(),
                                 Horizon=numeric(),
                                 Columns=numeric(),
                                 WAPE=numeric(),
                                 MAPE=numeric(),
                                 SMAPE=numeric(),
                                 stringsAsFactors = FALSE)

raw_mse_matrix <- c()

for(k in components){
   cumulative_X_hat <- list()

  for (dimensionality_method in dimensionality_methods){

    cat("[INFO] - Running dimensionality model: ",dimensionality_method,"\n")
    cumulative_X <- numeric()
    cumulative_X_hat[[dimensionality_method]] <- list()

      for(horizon in horizons){
        
        #Rolling window cross validation
        splitting_points <- round(seq(window_size,nrow(X)-horizon,by=window_size),0)
        
        #Rolling origin cross validation
        #splitting_points <- round(seq(splitting_point,by=horizon,length.out=cv_folds),0)
        
        for (i in 1:length(splitting_points)){
          
          cat("[INFO] - Running Rolling Window Fold ",i,"\n")
          X_train <- X[max((splitting_points[i]-window_size+1),1):splitting_points[i],1:n_series]
          if(perform_local_scale){
            X_train_scaled <- scale(X_train)
          }
          X_test <- X[(splitting_points[i]+1):(splitting_points[i]+horizon),1:n_series]
          
          #cat("[INFO] - Running Rolling Origin Fold ",i,"\n")
          # X_train <- X[1:splitting_points[i],1:n_series]
          # X_train_scaled <- scale(X_train)
          # X_test <- X[(splitting_points[i]+1):(splitting_points[i]+horizon),1:n_series]
          
          cumulative_X <- rbind(cumulative_X,X_test)
          
          # Univariate benchmarks
          # M4_benchmarks
          if(perform_local_scale){
            M4_results_list <-multivariate_M4benchmarks(X_train_scaled,horizon)
          }
          else{
            M4_results_list <-multivariate_M4benchmarks(X_train,horizon)
          }
          n_parameters <- 0
          
          # Naive
          forecasting_method <- "UniNaive"
          if(perform_local_scale){
            Xhat_naive <- unscale(M4_results_list$Naive,X_train_scaled)
          }
          else{
            Xhat_naive <- M4_results_list$Naive
          }
          forecast_time <- M4_results_list$TimeNaive
          MSE_naive <- MMSE(X_test,Xhat_naive)
          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=NA,
                                                ForecastingMethod=forecasting_method,
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(forecast_time), # Elapsed time
                                                MSE=as.numeric(MSE_naive$mean),
                                                NNMSE=as.numeric(MSE_naive$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
          cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind(cumulative_X_hat[[dimensionality_method]][[forecasting_method]],
                                                                                    Xhat_naive)
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_naive$values_vec)
          
          # Naive Seasonal
          forecasting_method <- "UniNaiveSeasonal"
          if(perform_local_scale){
            Xhat_naive_seasonal <- unscale(M4_results_list$NaiveSeasonal,X_train_scaled)
          }
          else{
            Xhat_naive_seasonal <- M4_results_list$NaiveSeasonal
          }
          forecast_time <- M4_results_list$TimeNaiveSeasonal
          MSE_naive_seasonal <- MMSE(X_test,Xhat_naive_seasonal)
          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=NA,
                                                ForecastingMethod=forecasting_method,
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(forecast_time), # Elapsed time
                                                MSE=as.numeric(MSE_naive_seasonal$mean),
                                                NNMSE=as.numeric(MSE_naive_seasonal$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
           cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],Xhat_naive_seasonal)
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_naive_seasonal$values_vec)
          
          # Naive 2
          forecasting_method <- "UniNaive2"
          if(perform_local_scale){
            Xhat_naive_2 <- unscale(M4_results_list$Naive2,X_train_scaled)
          }
          else{
            Xhat_naive_2 <- M4_results_list$Naive2
          }
          forecast_time <- M4_results_list$TimeNaive2
          MSE_naive_2 <- MMSE(X_test,Xhat_naive_2)
          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=NA,
                                                ForecastingMethod=forecasting_method,
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(forecast_time), # Elapsed time
                                                MSE=as.numeric(MSE_naive_2$mean),
                                                NNMSE=as.numeric(MSE_naive_2$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
           cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],Xhat_naive_2)
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_naive_2$values_vec)
          
          # Simple ES
          forecasting_method <- "UniSimpleES"
          if(perform_local_scale){
            Xhat_simple_es <- unscale(M4_results_list$SimpleES,X_train_scaled)
          }
          else{
            Xhat_simple_es <- M4_results_list$SimpleES
          }
          forecast_time <- M4_results_list$TimeSimpleES
          MSE_simple_es <- MMSE(X_test,Xhat_simple_es)
          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=NA,
                                                ForecastingMethod=forecasting_method,
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(forecast_time), # Elapsed time
                                                MSE=as.numeric(MSE_simple_es$mean),
                                                NNMSE=as.numeric(MSE_simple_es$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
           cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],Xhat_simple_es)
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_simple_es$values_vec)
          
          # Holt Winters
          forecasting_method <- "UniHoltWinters"
          if(perform_local_scale){
            Xhat_holt_winters <- unscale(M4_results_list$HoltWinters,X_train_scaled)
          }
          else{
            Xhat_holt_winters <- M4_results_list$HoltWinters
          }
          
          forecast_time <- M4_results_list$TimeHoltWinters
          MSE_holt_winters <- MMSE(X_test,Xhat_holt_winters)
          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=NA,
                                                ForecastingMethod=forecasting_method,
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(forecast_time), # Elapsed time
                                                MSE=as.numeric(MSE_holt_winters$mean),
                                                NNMSE=as.numeric(MSE_holt_winters$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
           cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],Xhat_holt_winters)
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_holt_winters$values_vec)
          
          # Holt Winters Damped
          forecasting_method <- "UniHoltWintersDamped"
          if(perform_local_scale){
            Xhat_holt_winters_damped <- unscale(M4_results_list$HoltWintersDamped,X_train_scaled)
          }
          else{
            Xhat_holt_winters_damped <- M4_results_list$HoltWintersDamped
          }
          forecast_time <- M4_results_list$TimeHoltWintersDamped
          MSE_holt_winters_damped <- MMSE(X_test,Xhat_holt_winters_damped)
          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=NA,
                                                ForecastingMethod=forecasting_method,
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(forecast_time), # Elapsed time
                                                MSE=as.numeric(MSE_holt_winters_damped$mean),
                                                NNMSE=as.numeric(MSE_holt_winters$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
           cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],Xhat_holt_winters_damped)
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_holt_winters_damped$values_vec)
          
          # Theta
          forecasting_method <- "Theta"
          if(perform_local_scale){
            Xhat_theta <- unscale(M4_results_list$Theta,X_train_scaled)
          }
          else{
            Xhat_theta <- M4_results_list$Theta
          }
          forecast_time <- M4_results_list$TimeTheta
          MSE_theta <- MMSE(X_test,Xhat_theta)
          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=NA,
                                                ForecastingMethod=forecasting_method,
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(forecast_time), # Elapsed time
                                                MSE=as.numeric(MSE_theta$mean),
                                                NNMSE=as.numeric(MSE_theta$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
           cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],Xhat_theta)
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_theta$values_vec)
          
          # Comb
          forecasting_method <- "UniComb"
          if(perform_local_scale){
            Xhat_comb <- unscale(M4_results_list$Comb,X_train_scaled)
          }
          else{
            Xhat_comb <- M4_results_list$Comb
          }
          forecast_time <- M4_results_list$TimeComb
          MSE_comb <- MMSE(X_test,Xhat_comb)
          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=NA,
                                                ForecastingMethod=forecasting_method,
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(forecast_time), # Elapsed time
                                                MSE=as.numeric(MSE_comb$mean),
                                                NNMSE=as.numeric(MSE_comb$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
           cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],Xhat_comb)
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_comb$values_vec)
          
          #for (i in 1:length(splitting_points)){
          #cat("[INFO] - Running Rolling Origin Fold ",i,"\n")
          #X_train <- X_scaled[1:splitting_points[i],1:n_series]
          #X_test <- X_scaled[(splitting_points[i]+1):(splitting_points[i]+horizon),1:n_series]
          # Fit the dimensionality model only once
          
          dim_params <- list()
          dim_params$method <- dimensionality_method
          dim_params$epochs <- epochs
          if(dimensionality_method %in% c("deep")){
            dim_params$deep_layers <- c(10,5,k)
          }
          if(dimensionality_method %in% c("deep_lstm","deep_gru")){
            dim_params$deep_layers <- c(10,k)
          }
          if(dimensionality_method != "PCA"){
            dim_family <- "Autoencoder_Keras"
          }
          else{
            dim_family <- "PCA"
          }

          if(dimensionality_method %in% ExtendedDFML::REQUIRE_EMBEDDING_METHODS){
            dim_params$time_window <- horizon
          }

          if(perform_local_scale){
            dim_red_results <- dimensionalityReduction(X_train_scaled,k,family=dim_family,dim_params)
          }
          else{
            dim_red_results <- dimensionalityReduction(X_train,k,family=dim_family,dim_params)
          }
          dim_params$model <- dim_red_results$model
          dim_params$time_dim <- dim_red_results$time_dim

          dim_inc_results <- dimensionalityIncrease(dim_red_results$Z,family=dim_family,model=dim_params$model,X_train,dim_params)
          if(nrow(X_train) == nrow(dim_inc_results$X_hat)){
            if(perform_local_scale){
              MSE_dimensionality <- ExtendedDFML::MMSE(X_train_scaled,dim_inc_results$X_hat)
            }
            else{
              MSE_dimensionality <- ExtendedDFML::MMSE(X_train,dim_inc_results$X_hat)
            }
          }
          else{
            if(perform_local_scale){
              MSE_dimensionality <- ExtendedDFML::MMSE(tail(X_train_scaled,nrow(dim_inc_results$X_hat)),dim_inc_results$X_hat)
            }
            else{
              MSE_dimensionality <- ExtendedDFML::MMSE(tail(X_train,nrow(dim_inc_results$X_hat)),dim_inc_results$X_hat)
            }
            
          }


          if(dimensionality_method == "PCA"){
            n_parameters <- prod(dim(dim_red_results$model$eigenvectors))
          }
          else{
            n_parameters <- dim_red_results$model$autoencoder$count_params()
          }

          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=dimensionality_method,
                                                ForecastingMethod="OnlyDim",
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(dim_red_results$time_dim), # Elapsed time
                                                MSE=as.numeric(MSE_dimensionality$mean),
                                                NNMSE=as.numeric(MSE_dimensionality$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_dimensionality$values_vec)

          for (forecasting_method in M4_METHODS) {
            forecast_params <- list()
            print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
            forecast_params$method <- forecasting_method
            
            if(perform_local_scale){
              results <- ExtendedDFML::DFML(X_train_scaled,
                                            dim_family,
                                            "M4Methods",
                                            dimensionality_parameters = dim_params,
                                            forecast_params,
                                            components,
                                            horizon)
              X_hat <- unscale(results$X_hat,X_train_scaled)
            }
            else{
              results <- ExtendedDFML::DFML(X_train,
                                            dim_family,
                                            "M4Methods",
                                            dimensionality_parameters = dim_params,
                                            forecast_params,
                                            components,
                                            horizon)
              X_hat <- results$X_hat
            }

             cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],X_hat)

            MSE_forecast <- ExtendedDFML::MMSE(X_test,X_hat)
            ss_results.df <- bind_rows(ss_results.df,
                                       data.frame(DimensionalityMethod=dimensionality_method,
                                                  ForecastingMethod=forecasting_method,
                                                  Parameters=as.numeric(n_parameters),
                                                  Dataset=dataset_name,
                                                  Components=as.numeric(k),
                                                  Horizon=as.numeric(horizon),
                                                  Columns=as.numeric(ncol(X)),
                                                  Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                                  MSE=as.numeric(MSE_forecast$mean),
                                                  NNMSE=as.numeric(MSE_forecast$mean)/as.numeric(MSE_naive$mean),
                                                  Samples=as.numeric(splitting_points[i])))
            
            raw_mse_matrix <- rbind(raw_mse_matrix,MSE_forecast$values_vec)
            
          } # EndFor Forecasting Methods - M4
          
          #browser()
          
          forecast_params <- list()
          forecast_params$m <- 3
          forecast_params$C <- 3
          forecast_params$Kmin <- 2
          forecast_params$FF <- 0
          
          for (forecasting_method in multistepAhead_methods) {
            print(paste("[INFO] - Testing",forecasting_method,"- h:",horizon))
            forecast_params$method <- forecasting_method
            
            if(perform_local_scale){
              results <- ExtendedDFML::DFML(X_train_scaled,
                                            dim_family,
                                            "multistepAhead",
                                            dimensionality_parameters = dim_params,
                                            forecast_params,
                                            components,
                                            horizon)
              X_hat <- unscale(results$X_hat,X_train_scaled)
            }
            else{
              results <- ExtendedDFML::DFML(X_train,
                                            dim_family,
                                            "multistepAhead",
                                            dimensionality_parameters = dim_params,
                                            forecast_params,
                                            components,
                                            horizon)
              X_hat <- results$X_hat
            }
            
             cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],X_hat)
            
            MSE_forecast <- ExtendedDFML::MMSE(X_test,X_hat)
            
            ss_results.df <- bind_rows(ss_results.df,
                                       data.frame(DimensionalityMethod=dimensionality_method,
                                                  ForecastingMethod=forecasting_method,
                                                  Parameters=as.numeric(n_parameters),
                                                  Dataset=dataset_name,
                                                  Components=as.numeric(k),
                                                  Horizon=as.numeric(horizon),
                                                  Columns=as.numeric(ncol(X)),
                                                  Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                                  MSE=as.numeric(MSE_forecast$mean),
                                                  NNMSE=as.numeric(MSE_forecast$mean)/as.numeric(MSE_naive$mean),
                                                  Samples=as.numeric(splitting_points[i])))
            
            raw_mse_matrix <- rbind(raw_mse_matrix,MSE_forecast$values_vec)
          }# EndFor Forecasting Methods - multistepAhead
          
          print(paste("[INFO] - Testing VAR","- h:",horizon))
          forecasting_method <- "VAR"
          forecast_params$method <- forecasting_method
          
          if(perform_local_scale){
            results <- ExtendedDFML::DFML(X_train_scaled,
                                          dim_family,
                                          "VAR",
                                          dimensionality_parameters = dim_params,
                                          forecast_params,
                                          components,
                                          horizon)
            X_hat <- unscale(results$X_hat,X_train_scaled)
          }
          else{
            results <- ExtendedDFML::DFML(X_train,
                                          dim_family,
                                          "VAR",
                                          dimensionality_parameters = dim_params,
                                          forecast_params,
                                          components,
                                          horizon)
            X_hat <- results$X_hat
          }
          
           cumulative_X_hat[[dimensionality_method]][[forecasting_method]] <- rbind( cumulative_X_hat[[dimensionality_method]][[forecasting_method]],X_hat)
          
          MSE_forecast <- ExtendedDFML::MMSE(X_test,X_hat)
          
          ss_results.df <- bind_rows(ss_results.df,
                                     data.frame(DimensionalityMethod=dimensionality_method,
                                                ForecastingMethod="VAR",
                                                Parameters=as.numeric(n_parameters),
                                                Dataset=dataset_name,
                                                Components=as.numeric(k),
                                                Horizon=as.numeric(horizon),
                                                Columns=as.numeric(ncol(X)),
                                                Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                                MSE=as.numeric(MSE_forecast$mean),
                                                NNMSE=as.numeric(MSE_forecast$mean)/as.numeric(MSE_naive$mean),
                                                Samples=as.numeric(splitting_points[i])))
          
          raw_mse_matrix <- rbind(raw_mse_matrix,MSE_forecast$values_vec)
          
          print(paste("[INFO] - Testing Gradient Boosting","- h:",horizon))
          
          forecast_params <- list()
          forecast_params$m <- 3
          forecast_params$forecasting_params <- list()
          forecast_params$forecasting_params$n_threads <- 8
          
          for (multistep_method in c("direct","recursive")){
            for (forecasting_method in GRADIENT_BOOSTING_METHODS) {
              print(paste("[INFO] - Testing",forecasting_method,multistep_method,"- h:",horizon))
              forecast_params$forecasting_method <- forecasting_method
              forecast_params$multistep_method <- multistep_method
              method_name <- paste(forecasting_method,multistep_method,sep="_")
              
              if(perform_local_scale){
                results <- ExtendedDFML::DFML(X_train_scaled,
                                              dim_family,
                                              "gradientBoosting",
                                              dimensionality_parameters = dim_params,
                                              forecast_params,
                                              components,
                                              horizon)
                X_hat <- unscale(results$X_hat,X_train_scaled)
              }
              else{
                results <- ExtendedDFML::DFML(X_train,
                                              dim_family,
                                              "gradientBoosting",
                                              dimensionality_parameters = dim_params,
                                              forecast_params,
                                              components,
                                              horizon)
                X_hat <- results$X_hat
              }
              
               cumulative_X_hat[[dimensionality_method]][[method_name]] <- rbind( cumulative_X_hat[[dimensionality_method]][[method_name]],X_hat)
              
              MSE_forecast <- ExtendedDFML::MMSE(X_test,X_hat)
              
              ss_results.df <- bind_rows(ss_results.df,
                                         data.frame(DimensionalityMethod=dimensionality_method,
                                                    ForecastingMethod=method_name,
                                                    Parameters=as.numeric(n_parameters),
                                                    Dataset=dataset_name,
                                                    Components=as.numeric(k),
                                                    Horizon=as.numeric(horizon),
                                                    Columns=as.numeric(ncol(X)),
                                                    Time=as.numeric(results$Time_dim+results$Time_forecast), # Elapsed time
                                                    MSE=as.numeric(MSE_forecast$mean),
                                                    NNMSE=as.numeric(MSE_forecast$mean)/as.numeric(MSE_naive$mean),
                                                    Samples=as.numeric(splitting_points[i])))
              
              raw_mse_matrix <- rbind(raw_mse_matrix,MSE_forecast$values_vec)
              
            }
          }
        } # EndFor CV
        
        article_df.list <- lapply(seq_along(cumulative_X_hat[[dimensionality_method]]), 
                                  function(i){
                                    computeMetrics(
                                    cumulative_X_hat[[dimensionality_method]][[i]], 
                                    cumulative_X,
                                    dimensionality_method,
                                    names(cumulative_X_hat[[dimensionality_method]])[i],
                                    n_parameters,
                                    dataset_name,
                                    components,
                                    horizon,
                                    as.numeric(ncol(X)))
                                  })
        
        article_results.df <- bind_rows(article_results.df,
                                        Reduce(rbind,article_df.list))
        
        saveRDS(ss_results.df,file.path(WORKDIR,paste(dataset_name,"_DFML","_dim","_results",".Rdata",sep = "")))
        saveRDS(article_results.df,file.path(WORKDIR,paste(dataset_name,"_article_results_df",".Rdata",sep = "")))
      } # EndFor Horizons
      # Clear Keras session to free up memory
      # Solved according https://github.com/keras-team/keras/issues/5345
      rm(dim_inc_results) # 1. Remove object containing model
      rm(dim_dec_results) # Remove object containing model
      k_clear_session() # 2. Tell Keras to free up the session
      py_gc$collect() # 3. Once the model is not referenced anymore and Keras has freed the session the garbage collector would work
  } # EndFor Dimensionality
}# EndFor Components


#Write final results
write.csv(ss_results.df,file.path(WORKDIR,paste(dataset_name,"_DFML","_dim","_results","_diff_scale",".csv",sep = "")),row.names = FALSE)
saveRDS(article_results.df,file.path(WORKDIR,paste(dataset_name,"_article_results_df","_diff_scale",".Rdata",sep = "")))
saveRDS(cumulative_X_hat,file.path(WORKDIR,paste(dataset_name,"_cumulative_X_hat","_diff_scale",".Rdata",sep = "")))
saveRDS(cumulative_X,file.path(WORKDIR,paste(dataset_name,"_cumulative_X","_diff_scale",".Rdata",sep = "")))
saveRDS(raw_mse_matrix,file.path(WORKDIR,paste(dataset_name,"_MSE_vec","_diff_scale",".Rdata",sep = "")))
