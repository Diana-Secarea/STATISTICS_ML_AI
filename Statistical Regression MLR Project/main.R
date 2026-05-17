#Using a specialised library to load the excel file
library(readxl)

#Import the data with the correct excel read approach
df <- read_excel("FINAL PROJECT CLEARED.xlsx", sheet = "Data", col_types = c("text", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric", "text"))

#verify the data
head(df)
head(df$`Population Density`)

#Data cleaning check
sum(is.na(df))
df <- na.omit(df)

#Renaming the columns
colnames(df)[2:7] <- c("Unemployment", "PopDensity", "AgriShare", "GDP", "Education", "CEE_Dummy")

#Descriptive statistics
summary(df[2:6])
sapply(df[2:6], sd)

#Full model, includes CEE Dummy
model_a <- lm(Unemployment ~ PopDensity + AgriShare + GDP + Education + CEE_Dummy, data = df)
summary(model_a)

#Restricted model, does not include CEE Dummy
model_b <- lm(Unemployment ~ PopDensity + AgriShare + GDP + Education, data = df)
summary(model_b)

#Check for the car library (Companion to Applied Regression)
if(!require(car)) install.packages("car")

#Checking Multicolliniarity
vif(model_a)

#dividing the window into 4 so we can see all 4 graphs at the same time
par(mfrow=c(2,2))

#generating the diagnostic charts
plot(model_a)

#Saphiro Wilk Test
shapiro.test(residuals(model_a))

#Load library for Breuch-Pagan test
if(!require(lmtest)) install.packages("lmtest")
library(lmtest)

#Homoscedasticity
bptest(model_a)
