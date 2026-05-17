#Libraries
  if(!require(readxl)) install.packages("readxl")
  if(!require(car)) install.packages("car")      # For VIF
  if(!require(lmtest)) install.packages("lmtest") # For BP and DW tests
  
  library(readxl)
  library(car)
  library(lmtest)
  
  getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
  }
#------------------------------------------------------------------------------------------------------------------------------
# Data import
  df <- read_excel("FINAL PROJECT CLEARED.xlsx", sheet = "Data", 
                   col_types = c("text", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "text"))
  colnames(df)[2:7] <- c("Unemployment", "PopDensity", "AgriShare", "GDP", "Education", "CEE_Dummy")
  
#------------------------------------------------------------------------------------------------------------------------------
# Descriptive
  print("DESCRIPTIVE STATISTICS: SUMMARY")
  summary(df[2:6])
  
  print("DESCRIPTIVE STATISTICS: MODE")
  sapply(df[2:6], getmode)

# Model A: Full model (Includes CEE_Dummy)
  model_a <- lm(Unemployment ~ PopDensity + AgriShare + GDP + Education + CEE_Dummy, data = df)
  print("---FULL MODEL (WITH DUMMY) RESULTS---")
  summary(model_a)
  
# Model B: Restricted model (No Dummy)
  model_b <- lm(Unemployment ~ PopDensity + AgriShare + GDP + Education, data = df)
  print("---RESTRICTED MODEL (WITHOUT DUMMY) RESULTS---")
  summary(model_b)
#------------------------------------------------------------------------------------------------------------------------------
print("---MODEL A---")
# A. Collinearity (Variance Inflation Factor)
  print("---COLLINEARITY---")
  vif(model_a)
  
# B. Normality of Errors (Shapiro-Wilk Test)
  print("---NORMALITY ---")
  shapiro.test(residuals(model_a))
  
# C. Homoscedasticity (Breusch-Pagan Test)
  print("---HOMOSCEDASTICITY---")
  bptest(model_a)
  
# D. Autocorrelation (Durbin-Watson Test)
  print("---AUTOCORRELATION---")
  dwtest(model_a)
#------------------------------------------------------------------------------------------------------------------------------
print("---MODEL B---")
  # A. Collinearity (Variance Inflation Factor)
  print("---COLLINEARITY---")
  vif(model_b)
  
  # B. Normality of Errors (Shapiro-Wilk Test)
  print("---NORMALITY ---")
  shapiro.test(residuals(model_b))
  
  # C. Homoscedasticity (Breusch-Pagan Test)
  print("---HOMOSCEDASTICITY---")
  bptest(model_b)
  
  # D. Autocorrelation (Durbin-Watson Test)
  print("---AUTOCORRELATION---")
  dwtest(model_b)
#------------------------------------------------------------------------------------------------------------------------------
# Visual confirmation of the tests above
  par(mfrow=c(2,2)) # Split view into 4 quadrants
  plot(model_a)
  par(mfrow=c(1,1)) # Reset view
  
  
