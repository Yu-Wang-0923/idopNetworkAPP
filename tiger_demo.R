S <- c(9.95, 24.45, 31.75, 35.00, 25.02, 16.86, 14.38, 9.60, 24.35, 27.50, 
       17.08, 37.00, 41.95, 11.66, 21.65, 17.89, 69.00, 10.30, 34.93, 46.59, 
       44.88, 54.12,  56.63, 22.13, 21.15)
L <- c(2, 8, 11, 10, 8, 4, 2, 2, 9, 8, 4, 11, 12, 
       2, 4, 4, 20, 1, 10, 15, 15, 16, 17, 6, 5)
H <- c(50, 110, 120, 550, 295, 200, 375, 52, 100, 300, 412, 400, 500, 
       360, 205,400, 600, 585, 540, 250, 290, 510, 590, 100, 400)
data1 <- data.frame(S = S, L = L, H = H)

library(orthopolynom)
data2 <- apply(data1, 2, scaleX, u=0,v=1)

total <- apply(data2,1,sum)
si_index <- total[order(total)]
data_exp_order <- data2[order(total),]

power_equation_fit2 <- function(x,y,X_smooth, thread = 4) {
  data <- y
  X <- x
  core.number <- thread
  cl <- makeCluster(getOption("cl.cores", core.number))
  clusterExport(cl, c("power.equation", "power_equation_base2", "data", "X","s.mle"), envir = environment())
  all_model = parLapply(cl = cl, 1:ncol(data), function(c) power_equation_base2(X, data[,c]))
  stopCluster(cl)
  
  
  names(all_model) = colnames(data)
  
  # new_x = seq(min(X), max(X), length = n)
  new_x = X
  new_x2 = X_smooth
  power_fit = sapply(1:ncol(data), function(c)power.equation(all_model[[c]],new_x))
  power_fit2 = sapply(1:ncol(data), function(c)power.equation(all_model[[c]],new_x2))
  colnames(power_fit) <- colnames(data)
  colnames(power_fit2) <- colnames(data)
  
  result = list(original_data = data, power_par = all_model, power_fit = power_fit,Time = X,smooth_fit = power_fit2)
  return(result)
}

power_equation_base2 <- function(x, y){
  x <- as.numeric(x)
  y <- as.numeric(y)
  return(optim(c(0.5,0.5),s.mle,data=y,x=x,method = "Nelder-Mead")$par)
}

power.equation <- function(par,x){
  par[1]*x^par[2]
}

s.mle<-function(par,data,x){
  y <- power.equation(par,x)
  yi <- data
  res <- sum((yi-y)^2)
  return(res)
}

library(parallel)
fit <- power_equation_fit2(
       x = si_index,
       y = data_exp_order,
       X_smooth = seq(si_index[1], si_index[length(si_index)], length.out = 25),
       thread = 3)

#plot fitting curves
tiff("F:/ppt-demo/points-plot.tiff",width=20,height=8,units = "cm",res=300)
par(mfrow=c(1,3),las=1)
par(mar = c(0, 0, 0,0),oma = c(4, 4.2, 2, 1),mgp=c(1,0.5,0))

col1 <- c("#FBB463","#88CEEA","#9BCC2F")
col2 <- c("#FDD2A3","#BDE3F4","#C8E38D")
for (i in 1:3) {
  x1 <- range(fit$Time)
  lim1 <- x1[2]-x1[1]
  plot(fit$Time,fit$original_data[,i],type = "n",ylab = "",xlab = "",axes=FALSE,
       ylim = c(-0.1,1.1),
       xlim = c(x1[1]-0.05*lim1,x1[2]+0.05*lim1))
  points(fit$Time,fit$original_data[,i],col=col2[i],pch=16,cex=1.5)
  lines(fit$Time,fit$power_fit[,i],col=col1[i],lwd=3)
  box(lwd=1)
  axis(1,at=seq(x1[1]+0.05*lim1,x1[2]-0.05*lim1,length=4),
       labels =sprintf("%0.1f", seq(x1[1]+0.05*lim1,x1[2]-0.05*lim1,length=4)),
       cex.axis=1.1)
  
  mtext(colnames(fit$original_data)[i],side = 3,line = 0.5,cex = 1.2)
  if(i==1){
    axis(2,at = seq(0,1,length=4),
         labels =sprintf("%0.2f", seq(0,1,length=4)),cex.axis=1.2)
  }
}
mtext("Subject Index",side = 1,line = 2,cex=1,outer = TRUE)
mtext("Phenotypic value",side = 2,line = 2.8,cex=1,outer = TRUE,las=0)

dev.off()


#' @title estimate parameters for LOP
#' @importFrom stats optim
#' @param effecti vector of observed data
#' @param effect matrix of fitted data
#' @param relationship list contain the result of lasso-based variable election
#' @param order_ind scalar of LOP order for independent part
#' @param order_dep scalar of LOP order for dependent parts
#' @return matrix of estimated LOP parameters
get_value <- function(effecti,effect,relationship,order_ind,order_dep){
  #input
  ind <- relationship[[1]]
  dep <- relationship[[2]]
  ind_no <- as.numeric(which(colnames(effect)==ind))
  dep_no <- as.numeric(sapply(1:length(dep), function(c) which(colnames(effect)==dep[c])))
  init_pars <- rep(0.001,(length(ind_no)*(order_ind+1)+length(dep_no)*(order_dep+1)))
  result <- optim(init_pars,ode_optimize,ind=ind_no,dep=dep_no,effecti=effecti,
                  data=effect,order_ind=order_ind,order_dep=order_dep,
                  method = "Nelder-Mead", control=list(maxit=50000,trace=T))
  par_after <- result$par
  return(par_after)
}

#' @title calculate least-square for observed and fitted data
#' @param pars matrix of LOP parameters for ind and dep growth curve
#' @param ind the independent growth curve id
#' @param dep the dependent growth curve id
#' @param effecti vector of observed data
#' @param data matrix of fitted data
#' @param order_ind scalar of LOP order for independent part
#' @param order_dep scalar of LOP order for dependent part
#' @return scalar of least-square error
ode_optimize <- function(pars,ind,dep,effecti,data,order_ind,order_dep){
  ind_pars <- pars[1:(order_ind+1)]
  dep_pars <- matrix(pars[-(1:(order_ind+1))],ncol=(order_dep+1))
  inital_value <- effecti[1]
  ind_effect <- get_effect(ind_pars,data[,ind],order_ind,inital_value)
  if (nrow(dep_pars)==1) {
    dep_effect <- get_effect(dep_pars,data[,dep],order_dep,0)
    y <- ind_effect+dep_effect
  }else{
    dep_effect <- sapply(1:length(dep), function(c)
      get_effect(dep_pars[c,],data[,dep[c]],order_dep,0))
    y <- ind_effect+rowSums(dep_effect)
  }
  ssr <- sum((effecti-y)^2)
  #add penalty
  alpha=length(which(ind_effect<0))
  return(ssr+alpha*sum(pmax(0,-ind_effect)))
}

#' @title generate a growth curve(ind or dep effect curve)
#' @param pars matrix of LOP parameters for ind and dep growth curve
#' @param effect scalar of observed genetic or any other data of a time point
#' @param order the order of LOP
#' @param y0 scalar of initial y
#' @return vector of generated curve
get_effect <- function(pars,effect,order,y0){
  #Legendre polynomials
  LOP <-  legendre.polynomials(order, normalized=F)
  #derivatives of LOP
  LOP_fit <-  sapply(1:length(pars),function(c) pars[c]*LOP[[c]])
  s_time <- scaleX(effect,u=-1,v=1)
  h <- diff(s_time) #per step h
  if(order==0){
    f <- function(x){LOP_fit}
  }else{
    f <- function(x){dy=do.call(sum,polynomial.values(polynomials=LOP_fit,x=x));dy}
  }
  dy_fit <- c(y0)
  for (j in 1:(length(effect)-1)) {
    k1 <- f(s_time[j])
    k2 <- f(s_time[j]+h[j]/2)
    k3 <- f(s_time[j]+h[j]/2)
    k4 <- f(s_time[j]+h[j])
    y <- dy_fit[j]+h[j]/6*(k1+2*(1-1/sqrt(2))*k2+2*(1+1/sqrt(2))*k3+k4)
    dy_fit <- c(dy_fit,y)
  }
  return(dy_fit)
}


relationship_all <- vector("list", 3)
relationship_all[[1]] <- list("S",c("L","H"))
relationship_all[[2]] <- list("L",c("S","H"))
relationship_all[[3]] <- list("H",c("S","L"))


data_clustered <- fit$power_fit
data_observed <- fit$original_data

library(pbapply)
order <- 6
fitall <- vector("list", order)
for (i in 1:order) {
  order_ind <- i-1
  order_dep <- i-1
  cl <- makeCluster(getOption("cl.cores", 8))
  clusterEvalQ(cl, {require(orthopolynom)})
  clusterExport(cl, c("data_clustered","order_ind","order_dep","relationship_all",
                      "get_value","ode_optimize","get_effect","data_observed"),
                envir=environment())
  lop_par <- pblapply(1:ncol(data_clustered),function(c)
    get_value(data_observed[,c],data_clustered,relationship_all[[c]],
              order_ind,order_dep),cl=cl)
  stopCluster(cl)
  fitall[[i]] <- lop_par
}

#' @title convert all ODE result into a list(use when multiply network needed)
#' @param input list contain the ODE solving result from get_ode_par function
#' @return list of output for all data
#' @export
get_all_net <- function(input){
  effect <- input$dataset
  effecti <- input$data_observed
  n = ncol(effect)
  par = input$lop_par
  order_ind = input$order_ind
  order_dep = input$order_dep
  relationship = input$relationship
  
  net <- lapply(1:n,function(c)
    get_ode_output(relationship = relationship[[c]],
                   par = par[[c]],
                   effecti=effecti,
                   effect = effect,
                   order_ind = order_ind,
                   order_dep = order_dep))
  return(net)
}

#' @title helper function of plot decomposition_curve
#' @param input1 list object from get_ode_par function
#' @param input2 list object from get_all_net function
#' @param i scalar of row number
#' @return datafrmae for plot
get_curve_data <- function(input1, input2, i){
  times <- input1$times
  cluster_mean <- input1$dataset
  d1 <- data.frame(input2[[i]][[6]],check.names = F)
  colnames(d1)[-1] <- input2[[i]][[2]]
  d1$sum <- rowSums(d1)
  d1$origin <- as.numeric(cluster_mean[,i])
  d1$x <- times
  return(d1)
}

#' @title helper function to convert ODE result
#' @param relationship list contain the result of lasso-based variable election
#' @param par vector conatin LOP parameters
#' @param effecti vector of observed data
#' @param effect matrix of observed data
#' @param order scalar of LOP order
#' @return list contain bunch of useful output result for a row
get_ode_output <- function(relationship, par, effecti, effect, order_ind,order_dep){
  #effect <- t(effect)
  output <- list()
  output[[1]] <- relationship[[1]]
  output[[2]] <- relationship[[2]]
  output[[3]] <- par[1:(order_ind+1)]
  output[[4]] <- matrix(par[-(1:(order_ind+1))],ncol=(order_dep+1))
  
  ind_no <- as.numeric(which(colnames(effect)==output[[1]]))
  dep_no <- as.numeric(sapply(1:length(output[[2]]),
                              function(c) which(colnames(effect)==output[[2]][c])))
  inital_value <- effecti[1]
  ind_effect <- get_effect(as.numeric(output[[3]]),effect[,ind_no],order_ind,inital_value)
  if (length(dep_no)==1) {
    dep_effect <- get_effect(as.numeric(output[[4]]),effect[,dep_no],order_dep,0)
  }else{
    dep_effect <- sapply(1:length(dep_no), function(c)
      get_effect(as.numeric(output[[4]][c,]),effect[,dep_no[c]],order_dep,0))
    colnames(dep_effect) <- dep_no
  }
  all_effect <- cbind(ind_effect,dep_effect)
  effect_mean <- apply(all_effect,2,mean)
  output[[5]] <- effect_mean
  output[[6]] <- all_effect
  return(output)
}

R2_table <- c()
for (o in 1:order){
  order_ind <- o-1
  order_dep <- o-1
  par_all_obj <- list(lop_par = fitall[[o]],
                      relationship = relationship_all,
                      data_observed=data_observed,
                      dataset = data_clustered,
                      times = si_index,
                      order_ind = order_ind,
                      order_dep = order_dep)
  module_net1 <- get_all_net(par_all_obj)
  R2 <- c()
  for (i in 1:length(relationship_all)) {
    fitted_values <- rowSums(module_net1[[i]][[6]])
    observed_values <- data_observed[,i]  # 替换为实际的观测变量
    sse <- sum((fitted_values-observed_values)^2)
    sst <- sum((observed_values-mean(observed_values))^2)
    R2_i <- 1-sse/sst
    R2 <- c(R2,R2_i)
  }
  R2_table <- cbind(R2_table,R2)
  colnames(R2_table)[o] <- o
}

max_order <- c()
for (i in 1:nrow(R2_table)) {
  max_order <- c(max_order,which.max(R2_table[i,]))
}


links_table <- c()
for (i in 1:length(relationship_all)) {
  nn <- max_order[i]
  order_ind <- nn-1
  order_dep <- nn-1
  
  d2 <- get_ode_output(relationship = relationship_all[[i]],
                       par = fitall[[nn]][[i]],
                       effecti=data_observed[,i],
                       effect = data_clustered,
                       order_ind = order_ind,
                       order_dep = order_dep)
  
  for (depi in 1:length(d2[[2]])) {
    links_table <- rbind(links_table,c(d2[[2]][depi],d2[[1]],
                                     as.numeric(d2[[5]][1]),
                                     as.numeric(d2[[5]][1+depi])))
  }
}


colnames(links_table)<-c("source","target","size","weight")
links_table<-as.data.frame(links_table)
links_table$edge_type<-ifelse(links_table$weight>0,1,2)
links_table$weight <- abs(as.numeric(links_table$weight))
rownames(links_table)<-rep(1:nrow(links_table))

write.csv(links_table, file = "F:/ppt-demo/links_table.csv", row.names = F,quote = F)
# The exported csv file can be passed into Cystoscape to draw a beautiful network diagram


#################

tiff("F:/ppt-demo/curve-decomposition.tiff",width=20,height=7,units = "cm",res=300)
par(mfrow=c(1,3),las=1)
par(mar = c(1.8, 2, 0,0),oma = c(2, 2, 2, 1),mgp=c(1,0.5,0))

fit1 <- power_equation_fit2(x = si_index, y = data_exp_order,
                            X_smooth=seq(si_index[1],si_index[length(si_index)],length.out= 50),thread = 4)
data_clustered <- fit1$smooth_fit
data_observed <- fit1$original_data
X_smooth=seq(si_index[1],si_index[length(si_index)],length.out= 50)


bgcol <- c(rgb(253,210,163,150,maxColorValue = 255),
           rgb(189,227,244,150,maxColorValue = 255),
           rgb(200,227,141,150,maxColorValue = 255))
for (i in 1:3) {
  
  nn <- max_order[i]
  order_ind <- nn-1
  order_dep <- nn-1
  
  d2 <- get_ode_output(relationship = relationship_all[[i]],
                       par = fitall[[nn]][[i]],
                       effecti=data_observed[,i],
                       effect = data_clustered,
                       order_ind = order_ind,
                       order_dep = order_dep)
  d1 <- data.frame(d2[[6]],check.names = F)
  colnames(d1)[-1] <- d2[[2]]
  d1$sum <- rowSums(d1)
  d1$x <- X_smooth
  nt <- length(d1$x)
  ind_name <- d2[[1]]
  
  yrang <- range(d1[,-ncol(d1)]);diffy <- yrang[2]-yrang[1]
  xrang <- range(d1$x);diffx <- xrang[2]-xrang[1]
  
  plot(d1$x,d1$sum,col="#0dceda",type = "n",lwd=2,ylab = "",xlab = "",axes=FALSE,
       ylim = c(yrang[1]-diffy*0.1,yrang[2]+diffy*0.1),
       xlim = c(xrang[1]-diffx*0.01,xrang[2]+diffx*0.15))
  rengin<-par("usr")  #提取坐标系范围
  rect(xleft=rengin[1],ybottom=rengin[3],xright=rengin[2],ytop=rengin[4],col=bgcol[i])
  # points(ei_index,data_observed[,i])
  lines(d1$x,d1$sum,col=rgb(0,153,255,maxColorValue = 255),lwd=2)
  lines(d1$x,d1$ind_effect,col="#E72041",lwd=2)
  for (d in 1:(ncol(d1)-3)) {
    lines(d1$x,d1[,1+d],col="#7DA449",lwd=2)
  }
  
  dep_name <- colnames(d1)[2:(ncol(d1)-2)][order(as.numeric(d1[nt,2:(ncol(d1)-2)]))]
  if(length(dep_name)==1){
    text(x=max(as.numeric(d1$x))+diffx*0.06, 
         y=d1[nt,2:(ncol(d1)-2)], 
         labels=dep_name,cex=1.3)
  }else if(length(dep_name)==2){
    text(x=rep(max(as.numeric(d1$x))+diffx*0.06, length(dep_name)), 
         y=seq(min(d1[nrow(d1),2:(ncol(d1)-2)]-0.1),
               max(d1[nrow(d1),2:(ncol(d1)-2)]+0.1),
               length=length(dep_name)), 
         labels=dep_name,cex=1.3)
  }else{
    text(x=rep(max(as.numeric(d1$x))+diffx*0.06, length(dep_name)), 
         y=seq(max((yrang[1]-diffy*0.05),min(d1[nrow(d1),2:(ncol(d1)-2)]-diffy*0.1)),
               min((yrang[2]+diffy*0.05),max(d1[nrow(d1),2:(ncol(d1)-2)]+diffy*0.1)),
               length=length(dep_name)), 
         labels=dep_name,cex=1.3)
  }
  
  box(lwd=1)
  abline(h=0,lty=2)
  axis(1,at=seq(xrang[1],xrang[2],length=4),
       labels =sprintf("%0.1f", seq(xrang[1],xrang[2],length=4)),cex.axis=1)
  axis(2,at = seq(yrang[1],yrang[2],length=4),
       labels =sprintf("%0.1f", seq(yrang[1],yrang[2],length=4)),cex.axis=1)
  mtext(colnames(fit$original_data)[i],side = 3,line = 0.5,cex = 1.2)
}
mtext("Subject Index",side = 1,line = 0.2,cex=1,outer = TRUE)
mtext("Phenotypic value",side = 2,line = 0.2,cex=1,outer = TRUE,las=0)

dev.off()
