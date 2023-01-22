##############################################################################################################
# AUTHOR: MACIEJ SLIZ-KONDRATOWICZ
# DATE: 28.07.2021
# CASE STUDY: Search for a model that utilizes US treasury rates to predict deposit rate paid
##############################################################################################################
##############################################################################################################
# PACKAGES
##############################################################################################################
library(xlsx)
library(dplyr)
library(zoo)
library(ggplot2)
library(lubridate)
library(plyr)
library(xts)
library(car)
library(devtools)
library(ggbiplot)
library(lmtest)
library(corrplot)
library(clusterSim)
library(factoextra)
library(pls)
library(tseries)
##############################################################################################################
# FUNCTIONS
##############################################################################################################
# Agregate US TY from daily data to quarter average, excluding NAs
aggregate_daily_to_quarterly <- function(inp){
  # inp - vector of daily series
  # Convert to xts
  inp.xts <- xts(x=inp, order.by=as.Date(inp$DATE))
  # Take only last column with US TY
  inp.xts = inp.xts[,-1]
  # Convert to numeric
  inp.xts.temp = unlist(lapply(1:length(inp.xts), function(n) as.numeric(inp.xts[n])))
  # To xts
  inp.xts.new <- xts(inp.xts.temp, order.by=as.Date(inp$DATE))
  # Omit NAs
  inp.xts.new = na.omit(inp.xts.new)
  # Create column name "QTRAVG"
  colnames(inp.xts.new) <- c("QTRAVG")
  # Aggregate to quarter average
  q.xts <- apply.quarterly(inp.xts.new,mean)
  # Define time index
  time(q.xts) <- as.yearqtr(time(q.xts))
  return(q.xts)
}

# Calculate RMSE
RMSE = function(m, o){
  sqrt(mean((m - o)^2))
}
##############################################################################################################
# Q1
# a): As you may notice, the data come with different frequency. 
# Please write code to clean the data and convert them to the desired frequency.
##############################################################################################################
setwd("...\\JPM - case\\data")

# READ IN INFORMATION ON DEPOSITS AND INTEREST EXPENSE
deposits <- read.xlsx("Deposit Interest.xlsx", sheetName = "Deposit Rate")

# READ IN HISTORICAL US TY
interest1M0   <- read.csv("DGS1MO.csv")
interest1     <- read.csv("DGS1.csv")
interest2     <- read.csv("DGS2.csv")
interest3M0   <- read.csv("DGS3MO.csv")
interest5     <- read.csv("DGS5.csv")
interest6M0   <- read.csv("DGS6MO.csv")
interest7     <- read.csv("DGS7.csv")
interest10    <- read.csv("DGS10.csv")
interest20    <- read.csv("DGS20.csv")
interest30    <- read.csv("DGS30.csv")

# AGGREGATE US TY (QTR AVERAGE)
interest1M0.qtravg <- aggregate_daily_to_quarterly(interest1M0)
interest1.qtravg <- aggregate_daily_to_quarterly(interest1)
interest2.qtravg <- aggregate_daily_to_quarterly(interest2)
interest3M0.qtravg <- aggregate_daily_to_quarterly(interest3M0)
interest5.qtravg <- aggregate_daily_to_quarterly(interest5)
interest6M0.qtravg <- aggregate_daily_to_quarterly(interest6M0)
interest7.qtravg <- aggregate_daily_to_quarterly(interest7)
interest10.qtravg <- aggregate_daily_to_quarterly(interest10)
interest20.qtravg <- aggregate_daily_to_quarterly(interest20)
interest30.qtravg <- aggregate_daily_to_quarterly(interest30)
# COMBINE ALL TENOR INTO ONE DATASET WITH X's
US.TY <- cbind.xts(interest1M0.qtravg,
                   interest1.qtravg,
                   interest2.qtravg,
                   interest3M0.qtravg,
                   interest5.qtravg,
                   interest6M0.qtravg,
                   interest7.qtravg,
                   interest10.qtravg,
                   interest20.qtravg,
                   interest30.qtravg)

colnames(US.TY) <- c("US Y 1M0",
                     "US Y 1Y",
                     "US Y 2Y",
                     "US Y 3M0",
                     "US Y 5Y",
                     "US Y 6M0",
                     "US Y 7Y",
                     "US Y 10Y",
                     "US Y 20Y",
                     "US Y 30Y")
# VISUALIZE HISTORICAL INTEREST RATE CURVE
# Set a color scheme:
tsRainbow <- rainbow(ncol(US.TY))
# Plot the overlayed series
plot.xts(x = US.TY, ylab = "US TY", main = "Historical US Treasury (in percentage points)",
     col = tsRainbow, screens = 1)
# Set a legend in the upper right hand corner to match color to return series
addLegend("topright", on=1, 
          legend.names = c("1 M0", "1Y", "2Y", "3 M0", "5Y", "6 M0", "7Y", "10Y", "20Y", "30Y"), 
          lty=c(1, 1), lwd=c(2, 1),
          col=tsRainbow)
##############################################################################################################
# Q1
# b): b): “Deposit Rate Paid” is not a provided variable. To get you started, 
# relevant data fields to calculate deposit rate paid by all FDIC insured banks 
# have been downloaded into the attached data file “Deposit Interest.xlsx”. 
# The column “Total interest expense” is the total interest paid in dollar amount. 
# There are a few related balance columns. Please choose the most appropriate balance 
# to use and discuss briefly why you think it is the appropriate one. 
# For the remaining questions, you can just use the “Deposit Rate Paid” variable as you defined here.
##############################################################################################################
# ANSWER:
# Deposit Rate is calculated as:
# Deposit Rate(t) = Total Interest Expense(t) / (Domestic deposits interest-bearing(t) + Foreign deposits(t))
# Domestic deposits non interest-bearing are not accounted for since these deopsits do not generate expense for the banks, 
# hence should not impact average deposit rate
# It is also assumed that all foreign deposits generate pay interest to the clients.

# Alternatives:
# - Split Interest paid into US and Foreign accounts, calculate two separate deposit rates
# - Use only Foreign accounts paying interest

# CREATE TIME INDEX
deposits.time.idx <- paste0(substr(deposits$YYYYQQ, 1, 4), " ", substr(deposits$YYYYQQ, 5, 6))
deposits <- as.xts(deposits[-1], order.by = as.yearqtr(deposits.time.idx))

# CREATE DEPENDENT VARIABLE
deposits$Y <- deposits$Total.interest.expens / (deposits$Domestic.deposits.interest.bearing + deposits$Foreign.deposits)
# ANNUALIZE DEPENDENT VARIABLE
deposits$Y <- (1 + deposits$Y) ^ (4) - 1
# MOVE FROM PERCENTAGE TO DECIMAL TO BE CONSISTENT WITH US TY
deposits$Y <- deposits$Y * 100

# Plot the overlayed series
plot.xts(x = deposits$Y , ylab = "Deposit rate", main = "Historical Annualized Deposit rate (in percentage points)",
         col = "red", screens = 1)
##############################################################################################################
# Q1
# c): (Bonus) Please write a gradient descent algorithm in python or R 
# to arrive at the coefficients for below linear regression:
#   <Deposit Rate Paid ~ TSY 1 Month>
##############################################################################################################
# COMBINE Y WITH X's
dataset <- cbind.xts(US.TY, deposits)
# VISUALIZE Y vs X
plot(na.omit(dataset$US.Y.1M0), type = "l", ylim = c(-0.5,10))
lines(na.omit(dataset$Y), type = "l", col = "red")

# OLS FORMULA: y(i) = theta_0(i) + theta_1(i) * x
# i - iteration
# INITIALIZE VECTORS OF THETA 0 AND THETA 1
theta0 <- c()
theta1 <- c()
theta0[1] <- 0
theta1[1] <- 1
# DEFINE LEARNING RATE
alpha <- -0.1
# DEFINE MAXIMUM ACCEPTABLE ERROR
min_grad <- 10^(-6)

# DEFINE Y AND X
Y <- dataset$Y
X <- dataset$US.Y.1M0

# INITATE GRADIENT DESCENT ALGORITHM
i = 0
eps <- c()
eps[1] <- sum((Y - (theta0[1] + theta1[1] * X))**2, na.rm = TRUE)
grad1 <- 1 / nrow(dataset) * sum((theta0[1] + theta1[1] * X - Y) * X, na.rm = TRUE)
while(abs(grad1) >= min_grad){
  i = i + 1
  grad0 <- 1 / nrow(dataset) * sum((theta0[i] + theta1[i] * X - Y), na.rm = TRUE)
  grad1 <- 1 / nrow(dataset) * sum((theta0[i] + theta1[i] * X - Y) * X, na.rm = TRUE)
  theta1[i + 1] <- theta1[i] + alpha * grad1
  theta0[i + 1] <- theta0[i] + alpha * grad0
  eps[i + 1] <- sum((Y - (theta0[i + 1] + theta1[i + 1] * X))**2, na.rm = TRUE)
}

# CHECK AGAINST COEFFICIENTS ESTIMATED WITH lm FUNCTION
model.check <- lm("Y ~ US.Y.1M0", data = dataset)
summary(model.check)

# PERCETAGE DIFFERENCE: THETA 0
100 * (theta0[length(theta0)] / model.check$coefficients[1] - 1)
# PERCETAGE DIFFERENCE: THETA 1
100 * (theta1[length(theta1)] / model.check$coefficients[2] - 1)
##############################################################################################################
# Q2 Modeling: by using the treasury rate, build a model to describe the deposit rate.
# Hint: Please feel free to use the best modeling methodology in your judgment. 
# While it’s important to understand the subject matter, for this project, please feel free to take as given 
# the target and explanatory variables as described above and work towards a reasonable statistical/Machine Learning 
# model first (please list the key steps and key criteria in your model selection), 
# and then add some discussions about the economic intuitions. If time permits, 
# you could but don’t have to explore beyond what’s provided.
##############################################################################################################
# ADD LAGS TO DATASET
dataset$Y.L1 <- lag.xts(dataset$Y, 1)
dataset$US.Y.1M0.L1 <- lag.xts(dataset$US.Y.1M0, 1)
dataset$US.Y.1Y.L1 <- lag.xts(dataset$US.Y.1Y, 1)
dataset$US.Y.2Y.L1 <- lag.xts(dataset$US.Y.2Y, 1)
dataset$US.Y.3M0.L1 <- lag.xts(dataset$US.Y.3M0, 1)
dataset$US.Y.5Y.L1 <- lag.xts(dataset$US.Y.5Y, 1)
dataset$US.Y.6M0.L1 <- lag.xts(dataset$US.Y.6M0, 1)
dataset$US.Y.7Y.L1 <- lag.xts(dataset$US.Y.7Y, 1)
dataset$US.Y.10Y.L1 <- lag.xts(dataset$US.Y.10Y, 1)
dataset$US.Y.20Y.L1 <- lag.xts(dataset$US.Y.20Y, 1)
dataset$US.Y.30Y.L1 <- lag.xts(dataset$US.Y.30Y, 1)

# ADD QoQ DIFFERENCES
dataset$Y.QoQ <- diff.xts(dataset$Y, 1)
dataset$US.Y.1M0.QoQ <- diff.xts(dataset$US.Y.1M0, 1)
dataset$US.Y.1Y.QoQ <- diff.xts(dataset$US.Y.1Y, 1)
dataset$US.Y.2Y.QoQ <- diff.xts(dataset$US.Y.2Y, 1)
dataset$US.Y.3M0.QoQ <- diff.xts(dataset$US.Y.3M0, 1)
dataset$US.Y.5Y.QoQ <- diff.xts(dataset$US.Y.5Y, 1)
dataset$US.Y.6M0.QoQ <- diff.xts(dataset$US.Y.6M0, 1)
dataset$US.Y.7Y.QoQ <- diff.xts(dataset$US.Y.7Y, 1)
dataset$US.Y.10Y.QoQ <- diff.xts(dataset$US.Y.10Y, 1)
dataset$US.Y.20Y.QoQ <- diff.xts(dataset$US.Y.20Y, 1)
dataset$US.Y.30Y.QoQ <- diff.xts(dataset$US.Y.30Y, 1)
dataset$Y.QoQ.L1 <- lag.xts(diff.xts(dataset$Y), 1)

# SCATTERPLOTS
scatterplotMatrix(~ Y + US.Y.1M0 + US.Y.1Y + US.Y.2Y + US.Y.3M0 + US.Y.5Y + US.Y.6M0 + US.Y.7Y + US.Y.10Y + US.Y.20Y + US.Y.30Y, data = dataset)

# SCATTERPLOTS - QoQ
scatterplotMatrix(~ Y.QoQ + US.Y.1M0.QoQ + US.Y.1Y.QoQ + US.Y.2Y.QoQ + US.Y.3M0.QoQ + US.Y.5Y.QoQ + US.Y.6M0.QoQ + US.Y.7Y.QoQ + US.Y.10Y.QoQ + US.Y.20Y.QoQ + US.Y.30Y.QoQ, data = dataset)

# CORRELATIONS - LEVELS
res <- cor(na.omit(dataset[,c('Y', 'US.Y.1M0', 'US.Y.1Y', 'US.Y.2Y', 'US.Y.3M0', 'US.Y.5Y', 'US.Y.6M0', 'US.Y.7Y', 'US.Y.10Y', 'US.Y.20Y', 'US.Y.30Y')]))
round(res, 2)

# CORRELATIONS - QoQ
res.QoQ <- cor(na.omit(dataset[,c('Y.QoQ', 'US.Y.1M0.QoQ', 'US.Y.1Y.QoQ', 'US.Y.2Y.QoQ', 'US.Y.3M0.QoQ', 'US.Y.5Y.QoQ', 'US.Y.6M0.QoQ', 'US.Y.7Y.QoQ', 'US.Y.10Y.QoQ', 'US.Y.20Y.QoQ', 'US.Y.30Y.QoQ')]))
round(res.QoQ, 2)

### OLS
# UNIT ROOT TESTING - DEPENDENT / INDEPENDENT VARIABLES
adf.test(na.omit(dataset$Y))

adf.test(na.omit(dataset$US.Y.1M0))
adf.test(na.omit(dataset$US.Y.3M0))
adf.test(na.omit(dataset$US.Y.6M0))
adf.test(na.omit(dataset$US.Y.1Y))
adf.test(na.omit(dataset$US.Y.2Y))
adf.test(na.omit(dataset$US.Y.5Y))
adf.test(na.omit(dataset$US.Y.10Y))
adf.test(na.omit(dataset$US.Y.20Y))
adf.test(na.omit(dataset$US.Y.30Y))

kpss.test(na.omit(dataset$Y))

kpss.test(na.omit(dataset$US.Y.1M0))
kpss.test(na.omit(dataset$US.Y.3M0))
kpss.test(na.omit(dataset$US.Y.6M0))
kpss.test(na.omit(dataset$US.Y.1Y))
kpss.test(na.omit(dataset$US.Y.2Y))
kpss.test(na.omit(dataset$US.Y.5Y))
kpss.test(na.omit(dataset$US.Y.10Y))
kpss.test(na.omit(dataset$US.Y.20Y))
kpss.test(na.omit(dataset$US.Y.30Y))

# OLS - QoQ
# REGRESS ALL US TY IN QoQ ON DEPOSIT RATE QoQ, EXCLUDE ONE BY ONE THE LEAST SIGNIFICANT ONES
model.1.1 <- lm("Y.QoQ ~ US.Y.1M0.QoQ + US.Y.1Y.QoQ + US.Y.2Y.QoQ + US.Y.3M0.QoQ + US.Y.5Y.QoQ + US.Y.6M0.QoQ + US.Y.7Y.QoQ + US.Y.10Y.QoQ + US.Y.20Y.QoQ + US.Y.30Y.QoQ", data = dataset)
summary(model.1.1)

model.1.2 <- lm("Y.QoQ ~ US.Y.1M0.QoQ + US.Y.1Y.QoQ + US.Y.2Y.QoQ + US.Y.3M0.QoQ + US.Y.5Y.QoQ + US.Y.6M0.QoQ + US.Y.7Y.QoQ + US.Y.10Y.QoQ + US.Y.30Y.QoQ", data = dataset)
summary(model.1.2)

model.1.3 <- lm("Y.QoQ ~ US.Y.1M0.QoQ + US.Y.1Y.QoQ + US.Y.2Y.QoQ + US.Y.3M0.QoQ + US.Y.5Y.QoQ + US.Y.6M0.QoQ + US.Y.7Y.QoQ + US.Y.30Y.QoQ", data = dataset)
summary(model.1.3)

model.1.4 <- lm("Y.QoQ ~ US.Y.1M0.QoQ + US.Y.2Y.QoQ + US.Y.3M0.QoQ + US.Y.5Y.QoQ + US.Y.6M0.QoQ + US.Y.7Y.QoQ + US.Y.30Y.QoQ", data = dataset)
summary(model.1.4)

model.1.5 <- lm("Y.QoQ ~ US.Y.2Y.QoQ + US.Y.3M0.QoQ + US.Y.5Y.QoQ + US.Y.6M0.QoQ + US.Y.7Y.QoQ + US.Y.30Y.QoQ", data = dataset)
summary(model.1.5)

model.1.6 <- lm("Y.QoQ ~ US.Y.2Y.QoQ + US.Y.3M0.QoQ + US.Y.6M0.QoQ + US.Y.7Y.QoQ + US.Y.30Y.QoQ", data = dataset)
summary(model.1.6)

model.1.7 <- lm("Y.QoQ ~ US.Y.2Y.QoQ + US.Y.3M0.QoQ + US.Y.6M0.QoQ + US.Y.7Y.QoQ", data = dataset)
summary(model.1.7)

model.1.8 <- lm("Y.QoQ ~ US.Y.2Y.QoQ + US.Y.6M0.QoQ + US.Y.7Y.QoQ", data = dataset)
summary(model.1.8)

model.1.9 <- lm("Y.QoQ ~ US.Y.2Y.QoQ + US.Y.6M0.QoQ", data = dataset)
summary(model.1.9)

### PCA - QoQ
# Splitting data into test and train data, will be needed later
dataset.QoQ <- dataset[,c('Y.QoQ', 'US.Y.1M0.QoQ', 'US.Y.3M0.QoQ', 'US.Y.6M0.QoQ', 'US.Y.1Y.QoQ', 'US.Y.2Y.QoQ', 'US.Y.5Y.QoQ', 'US.Y.7Y.QoQ','US.Y.10Y.QoQ','US.Y.20Y.QoQ', 'US.Y.30Y.QoQ')]
# REMOVE 30Y US TY DUE TO GAP IN 2002 - 2005
dataset.QoQ <- dataset.QoQ[,-which(colnames(dataset.QoQ) == 'US.Y.30Y.QoQ')]
# SPLIT: 90% TRAIN OF Y / 10% TEST
nrow(na.omit(dataset.QoQ$Y.QoQ))
start.Y.QoQ <- time(na.omit(dataset.QoQ$Y)[1])
end.train.QoQ <- as.yearqtr(start.Y + floor(nrow(na.omit(dataset.QoQ$Y)) * 0.9) / 4)
dataset.train.QoQ <- window(dataset.QoQ, end = as.yearqtr(end.train.QoQ))
dataset.test.QoQ <- window(dataset.QoQ, start = as.yearqtr(end.train.QoQ) + .25)
# Extracting the dependent variable y QoQ and removing it from the original dataframe.
depositrate.Y.QoQ <- dataset.train.QoQ$Y.QoQ
dataset.train.QoQ$Y.QoQ <- NULL
# Significant correlation between most of the variables.
res.QOQ <- cor(na.omit(dataset.train.QoQ), method="pearson")
corrplot::corrplot(res.QOQ, method= "color", order = "hclust", tl.pos = 'n')
# Normalisation before PCA
dataset.train.norm.QoQ <- data.Normalization(na.omit(dataset.train.QoQ), type="n1", normalization="column")
dataset.train.y.norm.QoQ <- data.Normalization(na.omit(depositrate.Y.QoQ), type="n1", normalization="column")
# PCA
depositrate.pca1.QoQ <- prcomp(dataset.train.norm.QoQ, center=TRUE, scale.=TRUE)
# OUTPUT
depositrate.pca1.QoQ$x [,1:9] %>% head(1) 
as.matrix(dataset.train.norm.QoQ) %*% as.matrix(depositrate.pca1.QoQ$rotation) [,1:9] %>% head(1)
# No correlation after pca
res1.QoQ <- cor(depositrate.pca1.QoQ$x, method="pearson")
corrplot::corrplot(res1.QoQ, method= "color", order = "hclust", tl.pos = 'n')
# Percentage of variance explained for each number of principal components 
plot(summary(depositrate.pca1.QoQ)$importance[3,], xlab = "Number of components", ylab = "Share of variabnce explained", lwd = 3, col = "blue")
abline(h = 0.99, col = "red", lwd = 3)
# Loading plots: how strongly a loading of a given variable contributes to a given principal component
# PC1-PC2
fviz_pca_var(depositrate.pca1.QoQ,axes = c(1, 2))
# PC3-PC4
fviz_pca_var(depositrate.pca1.QoQ,axes = c(3, 4))
# PC5-PC6
fviz_pca_var(depositrate.pca1.QoQ,axes = c(5, 6))
# PC7-PC8
fviz_pca_var(depositrate.pca1.QoQ,axes = c(7, 8))
# Plot PC1 per variable
plot(depositrate.pca1.QoQ$rotation[,"PC1"], type = "l", ylim = c(-0.5, 1), xaxt = "n", lwd = 3, col = "blue", ylab = "PC1", xlab = "X")
lines(depositrate.pca1.QoQ$rotation[,"PC1"] + depositrate.pca1.QoQ$sdev, type = "l", lwd = 3, col = "green")
lines(depositrate.pca1.QoQ$rotation[,"PC1"] - depositrate.pca1.QoQ$sdev, type = "l", lwd = 3, col = "green")
axis(1, at=1:9, labels = rownames(depositrate.pca1.QoQ$rotation))
abline(h = 0, col = "black", lwd = 3)
grid (10,10, lty = 6, col = "cornsilk2")
# Plot PC2 per variable
plot(depositrate.pca1.QoQ$rotation[,"PC2"], type = "l", ylim = c(-0.5, 1), xaxt = "n", lwd = 3, col = "blue", ylab = "PC2", xlab = "X")
lines(depositrate.pca1.QoQ$rotation[,"PC2"] + depositrate.pca1.QoQ$sdev, type = "l", lwd = 3, col = "green")
lines(depositrate.pca1.QoQ$rotation[,"PC2"] - depositrate.pca1.QoQ$sdev, type = "l", lwd = 3, col = "green")
axis(1, at=1:9, labels = rownames(depositrate.pca1.QoQ$rotation))
abline(h = 0, col = "black", lwd = 3)
grid (10,10, lty = 6, col = "cornsilk2")
# Plot PC3 per variable
plot(depositrate.pca1.QoQ$rotation[,"PC3"], type = "l", ylim = c(-0.5, 1), xaxt = "n", lwd = 3, col = "blue", ylab = "PC3", xlab = "X")
lines(depositrate.pca1.QoQ$rotation[,"PC3"] + depositrate.pca1.QoQ$sdev, type = "l", lwd = 3, col = "green")
lines(depositrate.pca1.QoQ$rotation[,"PC3"] - depositrate.pca1.QoQ$sdev, type = "l", lwd = 3, col = "green")
axis(1, at=1:9, labels = rownames(depositrate.pca1.QoQ$rotation))
abline(h = 0, col = "black", lwd = 3)
grid (10,10, lty = 6, col = "cornsilk2")
# Plot PC4 per variable
plot(depositrate.pca1.QoQ$rotation[,"PC4"], type = "l", ylim = c(-0.5, 1), xaxt = "n", lwd = 3, col = "blue", ylab = "PC4", xlab = "X")
lines(depositrate.pca1.QoQ$rotation[,"PC4"] + depositrate.pca1.QoQ$sdev, type = "l", lwd = 3, col = "green")
lines(depositrate.pca1.QoQ$rotation[,"PC4"] - depositrate.pca1.QoQ$sdev, type = "l", lwd = 3, col = "green")
axis(1, at=1:9, labels = rownames(depositrate.pca1.QoQ$rotation))
abline(h = 0, col = "black", lwd = 3)
grid (10,10, lty = 6, col = "cornsilk2")
# PCs vs Y
# Scatterplots
# PC1 
pcs.QoQ <- as.data.frame(depositrate.pca1.QoQ$x)
plot(as.numeric(window(dataset.train.y.norm.QoQ, start = as.yearqtr("2001 Q4"))), pcs.QoQ$PC1)
# PC2
plot(as.numeric(window(dataset.train.y.norm.QoQ, start = as.yearqtr("2001 Q4"))), pcs.QoQ$PC2)
# PC3
plot(as.numeric(window(dataset.train.y.norm.QoQ, start = as.yearqtr("2001 Q4"))), pcs.QoQ$PC3)
# PCR
ols.data.QoQ <- cbind.xts(window(dataset.train.y.norm.QoQ, start = as.yearqtr("2001 Q4")), as.xts(pcs.QoQ, order.by = time(window(dataset.train.y.norm.QoQ, start = as.yearqtr("2001 Q4")))))
# lm function for linear regression. 
lmodel.QoQ <- lm(Y.QoQ ~ ., data = ols.data.QoQ)
summary(lmodel.QoQ)
# MODEL WITH PC1-3
lmodel.QoQ.2 <- lm(Y.QoQ ~ PC1 + PC2 + PC3, data = ols.data.QoQ)
summary(lmodel.QoQ.2)
# Estimated coefficients multiplied by matrix V to obtain betas to be used in prediction
beta.Z.QoQ <- as.matrix(lmodel.QoQ.2$coefficients[2:4])
V.QoQ <- as.matrix(depositrate.pca1.QoQ$rotation)
beta.X.QoQ <- V.QoQ[,1:3] %*% beta.Z.QoQ
beta.X.QoQ
# results of linear regression performed on data using PCA and without it. 
lmodel.none.QoQ <- lm(Y.QoQ ~ ., data = window(dataset.QoQ, start = as.yearqtr("2001 Q4")))
summary(lmodel.none.QoQ)
# Oos result
# Data Normalization
dataset.test.norm.QoQ <- data.Normalization(window(dataset.test.QoQ[,-1], end = as.yearqtr("2017 Q4")))
dataset.test.y.norm.QoQ <- data.Normalization(window(dataset.test.QoQ[,1], end = as.yearqtr("2017 Q4")))
y.pred.test1.QoQ <- lmodel.QoQ.2$coefficients[1] + dataset.test.norm.QoQ %*% beta.X.QoQ
# OLS UP TO 2014 Q2
model.1.9.upto.2014Q2 <- lm("Y.QoQ ~ US.Y.2Y.QoQ + US.Y.6M0.QoQ", data = window(dataset, start = as.yearqtr("2001 Q4"), end = as.yearqtr("2014 Q2")))
summary(model.1.9.upto.2014Q2)
model.1.9.pred.QoQ <- predict(model.1.9.upto.2014Q2,newdata=dataset.test.norm.QoQ)
# PLOT FITTED VS ACTUALS - OLS VS PCA
plot(as.numeric(dataset.test.y.norm.QoQ), type = "l", lwd = 3, ylim = c(-0.15, 0.25), xaxt = "n", xlab = "Time", ylab = "Deposit Rate, QoQ, percentage points")
lines(y.pred.test1.QoQ, type = "l", col = "red", lwd = 3)
lines(model.1.9.pred.QoQ, type = "l", col = "blue", lwd = 3)
axis(1, at=1:14, labels = time(dataset.test.norm.QoQ))
abline(h = 0, col = "black", lwd = 3)
grid (10,10, lty = 6, col = "cornsilk2")
legend("topleft", c("PCA", "OLS"), 
          lty=c(1, 1), lwd=c(2, 1),
          col=c("red","blue"))
# RMSE
# PCA
RMSE(dataset.test.y.norm.QoQ, y.pred.test1.QoQ)
# OLS
RMSE(dataset.test.y.norm.QoQ, model.1.9.pred.QoQ)
