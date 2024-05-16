library(readr)
library(tidyverse)
library(xts)
library(forecast)
library(MLmetrics)

#import data and preprocess----
data <- read_csv("owid-covid-data.csv", col_types = cols(date = col_date(format = "%d/%m/%Y")))

#overview of data
glimpse(data)

#extract Japan data only
japan_data <- data %>%
  filter(location == "Japan")

#replace NA with 0 for columns 'total_cases' and 'total_deaths'
japan_data <- japan_data %>%
  replace_na(list(total_deaths = 0, total_cases = 0))
  

#find the date of the first confirmed case in Japan
start_date <- min(japan_data$date[japan_data$total_cases > 0]) #2020-01-14

prediction_data <- japan_data %>%
  select(date, new_cases, total_cases, new_deaths, total_deaths) %>% #select columns for prediction
  filter(date >= start_date & date <= "2022-12-31")


#convert to time series object and train-test split
timeData <- xts(prediction_data[, 2:5], order.by = prediction_data$date) #convert to time series data

#beginning to 2022-12-31
timeDataTrain <- window(timeData, start = "2020-01-14", end = "2022-10-02") 
timeDataTest <- window(timeData, start = "2022-10-03", end = "2022-12-31") #get last 90 days for testing, since we are forecasting 3 new month

#split into 2 categories
newCasesTrain <- timeDataTrain[, 1] #training for new cases
newCasesTest <- timeDataTest[, 1] #testing for new cases

newDeathsTrain <- timeDataTrain[, 3] #training for new deaths
newDeathsTest <- timeDataTest[, 3] #testing for new deaths

# ---- MODEL FITTING & EVALUATION ----
#----------- State Space Models (ETS)-----------
newCasesEts <- ets(newCasesTrain)
newDeathsEts <- ets(newDeathsTrain)


# Calculate MAPE
newCasesEtsMape <- MAPE(as.data.frame(forecast(newCasesEts, h=90))$`Point Forecast`[newCasesTest != 0], newCasesTest[newCasesTest != 0]) * 100 #exclude zero values
newDeathsEtsMape <- MAPE(as.data.frame(forecast(newDeathsEts, h=90))$`Point Forecast`[newDeathsTest != 0], newDeathsTest[newDeathsTest != 0]) * 100 #exclude zero values

# Calculate RMSE
newCasesEtsRMSE <- sqrt(mean((as.data.frame(forecast(newCasesEts, h=90))$`Point Forecast` - newCasesTest)^2, na.rm = TRUE))
newDeathsEtsRMSE <- sqrt(mean((as.data.frame(forecast(newDeathsEts, h=90))$`Point Forecast` - newDeathsTest)^2, na.rm = TRUE))

# Calculate MSE
newCasesEtsMSE <- mean((as.data.frame(forecast(newCasesEts, h=90))$`Point Forecast` - newCasesTest)^2, na.rm = TRUE)
newDeathsEtsMSE <- mean((as.data.frame(forecast(newDeathsEts, h=90))$`Point Forecast` - newDeathsTest)^2, na.rm = TRUE)

# Calculate MAE
newCasesEtsMAE <- mean(abs(as.data.frame(forecast(newCasesEts, h=90))$`Point Forecast` - newCasesTest), na.rm = TRUE)
newDeathsEtsMAE <- mean(abs(as.data.frame(forecast(newDeathsEts, h=90))$`Point Forecast` - newDeathsTest), na.rm = TRUE)

# Calculate R^2
newCasesEtsRSquared <- 1 - sum((newCasesTest - as.data.frame(forecast(newCasesEts, h=90))$`Point Forecast`)^2) / sum((newCasesTest - mean(newCasesTest))^2)
newDeathsEtsRSquared <- 1 - sum((newDeathsTest - as.data.frame(forecast(newDeathsEts, h=90))$`Point Forecast`)^2) / sum((newDeathsTest - mean(newDeathsTest))^2)



#--------------TBATS--------------
newCasesTbats <- tbats(newCasesTrain)
newDeathsTbats <- tbats(newDeathsTrain)

# Exclude zero values and calculate MAPE
newCasesTbatsMape <- MAPE(as.data.frame(forecast(newCasesTbats, h=90))$`Point Forecast`[newCasesTest != 0], newCasesTest[newCasesTest != 0]) * 100
newDeathsTbatsMape <- MAPE(as.data.frame(forecast(newDeathsTbats, h=90))$`Point Forecast`[newDeathsTest != 0], newDeathsTest[newDeathsTest != 0]) * 100

# Calculate RMSE
newCasesTbatsRMSE <- sqrt(mean((as.data.frame(forecast(newCasesTbats, h=90))$`Point Forecast` - newCasesTest)^2, na.rm = TRUE))
newDeathsTbatsRMSE <- sqrt(mean((as.data.frame(forecast(newDeathsTbats, h=90))$`Point Forecast` - newDeathsTest)^2, na.rm = TRUE))

# Calculate MSE
newCasesTbatsMSE <- mean((as.data.frame(forecast(newCasesTbats, h=90))$`Point Forecast` - newCasesTest)^2, na.rm = TRUE)
newDeathsTbatsMSE <- mean((as.data.frame(forecast(newDeathsTbats, h=90))$`Point Forecast` - newDeathsTest)^2, na.rm = TRUE)

# Calculate MAE
newCasesTbatsMAE <- mean(abs(as.data.frame(forecast(newCasesTbats, h=90))$`Point Forecast` - newCasesTest), na.rm = TRUE)
newDeathsTbatsMAE <- mean(abs(as.data.frame(forecast(newDeathsTbats, h=90))$`Point Forecast` - newDeathsTest), na.rm = TRUE)

# Calculate R^2
newCasesTbatsRSquared <- 1 - sum((newCasesTest - as.data.frame(forecast(newCasesTbats, h=90))$`Point Forecast`)^2) / sum((newCasesTest - mean(newCasesTest))^2)
newDeathsTbatsRSquared <- 1 - sum((newDeathsTest - as.data.frame(forecast(newDeathsTbats, h=90))$`Point Forecast`)^2) / sum((newDeathsTest - mean(newDeathsTest))^2)


#--------------HOLT'S TREND METHOD--------------
# Fit the models
newCasesHolt <- holt(newCasesTrain, h=90)
newDeathsHolt <- holt(newDeathsTrain, h=90)

# Exclude zero values and calculate MAPE
newCasesHoltMape <- MAPE(newCasesHolt$mean[newCasesTest != 0], newCasesTest[newCasesTest != 0]) * 100
newDeathsHoltMape <- MAPE(newDeathsHolt$mean[newDeathsTest != 0], newDeathsTest[newDeathsTest != 0]) * 100

# Calculate RMSE
newCasesHoltRMSE <- sqrt(mean((newCasesHolt$mean - coredata(newCasesTest))^2, na.rm = TRUE))  #coredata is added to the ('newCasesTest' and 'newDeathsTest') to convert xts object to numeric vector so that can subtact using same data type
newDeathsHoltRMSE <- sqrt(mean((newDeathsHolt$mean - coredata(newDeathsTest))^2, na.rm = TRUE))

# Calculate MSE
newCasesHoltMSE <- mean((newCasesHolt$mean - coredata(newCasesTest))^2, na.rm = TRUE)
newDeathsHoltMSE <- mean((newDeathsHolt$mean - coredata(newDeathsTest))^2, na.rm = TRUE)

# Calculate MAE
newCasesHoltMAE <- mean(abs(newCasesHolt$mean - coredata(newCasesTest)), na.rm = TRUE)
newDeathsHoltMAE <- mean(abs(newDeathsHolt$mean - coredata(newDeathsTest)), na.rm = TRUE)

# Calculate R^2
newCasesHoltRSquared <- 1 - sum((coredata(newCasesTest) - newCasesHolt$mean)^2) / sum((newCasesTest - mean(newCasesTest))^2)
newDeathsHoltRSquared <- 1 - sum((coredata(newDeathsTest) - newDeathsHolt$mean)^2) / sum((newDeathsTest - mean(newDeathsTest))^2)


#--------------ARIMA--------------
# Fit the ARIMA models
newCasesArima <- auto.arima(newCasesTrain)
newDeathsArima <- auto.arima(newDeathsTrain)

# Forecast 90 steps ahead
newCasesArimaForecast <- forecast(newCasesArima, h=90)
newDeathsArimaForecast <- forecast(newDeathsArima, h=90)

# Exclude zero values and calculate MAPE
newCasesArimaMape <- MAPE(newCasesArimaForecast$mean[newCasesTest != 0], newCasesTest[newCasesTest != 0]) * 100
newDeathsArimaMape <- MAPE(newDeathsArimaForecast$mean[newDeathsTest != 0], newDeathsTest[newDeathsTest != 0]) * 100

# Calculate RMSE
newCasesArimaRMSE <- sqrt(mean((newCasesArimaForecast$mean - coredata(newCasesTest))^2, na.rm = TRUE))
newDeathsArimaRMSE <- sqrt(mean((newDeathsArimaForecast$mean - coredata(newDeathsTest))^2, na.rm = TRUE))

# Calculate MSE
newCasesArimaMSE <- mean((newCasesArimaForecast$mean - coredata(newCasesTest))^2, na.rm = TRUE)
newDeathsArimaMSE <- mean((newDeathsArimaForecast$mean - coredata(newDeathsTest))^2, na.rm = TRUE)

# Calculate MAE
newCasesArimaMAE <- mean(abs(newCasesArimaForecast$mean - coredata(newCasesTest)), na.rm = TRUE)
newDeathsArimaMAE <- mean(abs(newDeathsArimaForecast$mean - coredata(newDeathsTest)), na.rm = TRUE)

# Calculate R^2
newCasesArimaRSquared <- 1 - sum((coredata(newCasesTest) - newCasesArimaForecast$mean)^2) / sum((newCasesTest - mean(newCasesTest))^2)
newDeathsArimaRSquared <- 1 - sum((coredata(newDeathsTest) - newDeathsArimaForecast$mean)^2) / sum((newDeathsTest - mean(newDeathsTest))^2)


#summarizing results
Models <- data.frame(
  Type = c("New cases", "New deaths", "New cases", "New deaths","New cases", "New deaths", "New cases", "New deaths"),
  Model = c("ETS", "ETS", "TBATS", "TBATS", "HOLTS", "HOLTS", "ARIMA","ARIMA"),
  MAPE = round(c(newCasesEtsMape, newDeathsEtsMape, newCasesTbatsMape, newDeathsTbatsMape, newCasesHoltMape, newDeathsHoltMape, newCasesArimaMape, newDeathsArimaMape), 5),
  RMSE = round(c(newCasesEtsRMSE, newDeathsEtsRMSE, newCasesTbatsRMSE, newDeathsTbatsRMSE, newCasesHoltRMSE, newDeathsHoltRMSE, newCasesArimaRMSE, newDeathsArimaRMSE), 5),
  MSE = round(c(newCasesEtsMSE, newDeathsEtsMSE, newCasesTbatsMSE, newDeathsTbatsMSE, newCasesHoltMSE, newDeathsHoltMSE, newCasesArimaMSE, newDeathsArimaMSE), 5),
  MAE = round(c(newCasesEtsMAE, newDeathsEtsMAE, newCasesTbatsMAE, newDeathsTbatsMAE, newCasesHoltMAE, newDeathsHoltMAE, newCasesArimaMAE, newDeathsArimaMAE), 5),
  RSquared = round(c(newCasesEtsRSquared, newDeathsEtsRSquared, newCasesTbatsRSquared, newDeathsTbatsRSquared, newCasesHoltRSquared, newDeathsHoltRSquared, newCasesArimaRSquared, newDeathsArimaRSquared), 5)
)

#forecast and plotting----
#best model = tbats
newCasesForecastTbats <- tbats(timeData$new_cases) %>% #fit tbats model on whole data for new cases
  forecast(h=90)  

newDeathsForecastTbats <- tbats(timeData$new_deaths) %>% #fit tbats model on whole data for new deaths
  forecast(h=90) 

newCasesForecast <- as.data.frame(newCasesForecastTbats)$`Point Forecast` #extract predicted value for new cases
newDeathsForecast <- as.data.frame(newDeathsForecastTbats)$`Point Forecast` #extract predicted value for new deaths

newForecast <- data.frame( 
  Date = seq(as.Date("2023-01-01"), as.Date("2023-03-31"), by = 1), #column for next 3 months
  New_cases = as.integer(newCasesForecast), #column for predicted values of new cases
  New_deaths = as.integer(newDeathsForecast) #column for predicted value of new deaths
) 
#%>%
#{.[-31, ]} #remove 2022-07-01 row

#plot for new cases
(newCasesPlot <- ggplot() + #current data
    geom_line(data = prediction_data, aes(x = date, y = new_cases, colour = "black")) +
    geom_line(data = newForecast, aes(x = Date, y = New_cases, colour = "red")) + #predicted data 
    labs(title = "New COVID-19 Cases of Japan from 2020-01-14 till 2023-03-31 (Predicted)",
         x = "Date",
         y = "New cases") +
    scale_color_manual(name = "", values = c("black","red"), labels = c("actual", "predicted")))

#zoom in on prediction
newCasesPlot + coord_cartesian(
  xlim = as.Date(c("2023-02-28", "2023-03-31")),
  ylim = c(0, 500))

#plot for new deaths
(newDeathsPlot <- ggplot() + #current data
    geom_line(data = prediction_data, aes(x = date, y = new_deaths, colour = "black")) +
    geom_line(data = newForecast, aes(x = Date, y = New_deaths, colour = "red")) + #predicted data
    labs(title = "New COVID-19 Deaths of Japan from 2020-01-14 till 2023-03-31 (Predicted)",
         x = "Date",
         y = "New deaths") +
    scale_color_manual(name = "", values = c("black","red"), labels = c("actual", "predicted")))


######NEW############
#extract historical data
historical_data <- prediction_data %>% 
  filter(date < as.Date("2022-10-03")) 


#forecast and plotting----
#best model = tbats
newCasesForecastTbats <- tbats(timeData$new_cases) %>% #fit tbats model on whole data for new cases
  forecast(h=90)  

newDeathsForecastTbats <- tbats(timeData$new_deaths) %>% #fit tbats model on whole data for new deaths
  forecast(h=90) 

newCasesForecast <- as.data.frame(newCasesForecastTbats)$`Point Forecast` #extract predicted value for new cases
newDeathsForecast <- as.data.frame(newDeathsForecastTbats)$`Point Forecast` #extract predicted value for new deaths

newForecast <- data.frame( 
  Date = seq(as.Date("2023-01-01"), as.Date("2023-03-31"), by = 1), #column for next 3 months
  New_cases = as.integer(newCasesForecast), #column for predicted values of new cases
  New_deaths = as.integer(newDeathsForecast) #column for predicted value of new deaths
) 

#plot for new cases
(newCasesPlot <- ggplot() +
    geom_line(data = historical_data, aes(x = date, y = new_cases, colour = "grey"), size = 1) + #historical data
    geom_line(data = prediction_data, aes(x = date, y = new_cases, colour = "black"), size = 1) + #actual data 
    geom_point(data = prediction_data, aes(x = date, y = new_cases, color = "black"), size = 1) + #actual data points
    geom_line(data = newForecast, aes(x = Date, y = New_cases, colour = "red"), size = 1, linetype = "longdash") + #predicted data 
    geom_point(data = newForecast, aes(x = Date, y = New_cases, color = "red"), size = 1) + #predicted data points
    labs(title = "New COVID-19 Cases of Japan from 2020-01-14 till 2023-03-31 (Predicted)",
         x = "Date",
         y = "New cases") +
    scale_color_manual(name = "", values = c("grey", "black", "red"), labels = c("historical", "actual", "predicted")))



