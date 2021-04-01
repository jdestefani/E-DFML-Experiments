# Multivariate error measures according to:
# According to https://arxiv.org/pdf/1905.03806.pdf

WAPE <- function(X,X_hat){
  num_vec <- apply(abs(X-X_hat), 2, sum)
  den_vec <- apply(abs(X),2,sum)
  wape_vec <- num_vec/den_vec
  return(list(values_vec=wape_vec,
              sum=sum(wape_vec,na.rm=T)))
}

MAPE <- function(X,X_hat){
  filter <- abs(X) > 0
  z0_vec <- apply(abs(X)>0, 2, sum)
  wape_results <- WAPE(X * filter,X_hat * filter)
  return(list(values_vec=wape_results$values_vec/z0_vec,
              sum=wape_results$sum/sum(z0_vec,na.rm = T)))
}

SMAPE <- function(X,X_hat){
  filter <- abs(X) > 0
  z0_vec <- apply(abs(X)>0, 2, sum)
  num_vec <- apply(abs(X-X_hat) * filter, 2, sum)
  den_vec <- apply(abs(X+X_hat),2,sum)
  smape_vec <- (2*num_vec)/den_vec
  return(list(values_vec=smape_vec/z0_vec,
              sum=sum(smape_vec,na.rm = T)/sum(z0_vec,na.rm = T)))
}
