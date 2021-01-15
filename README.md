# Rental_Bike_Prediction

## Overview

- Built a regression model to predict rental bike demand 
- Conducted feature engineering to generate features from date and time
- Exploratory data analysis on average user counts over time
- Tried 2 stepwise regression models, 1 on normal response variable and another on log transformed response variable
- Log transformed model achieved higher R squared of 0.80

## Tools Used

- Language: R
- Packages: lubridate, xts, ggplot2, gridExtra, corrplot, caret, MASS, dplyr, Metrics
- Data: [Kaggle](https://www.kaggle.com/c/bike-sharing-demand)
- Topics: R, Regression, Stepwise Regression

## Data 

The data is taken from [Kaggle](https://www.kaggle.com/c/bike-sharing-demand/overview). It contains the two-year historical log corresponding to years 2011 and 2012 from Capital Bikeshare system, Washington D.C., USA. Details of its columns are as follows:

| Variable   | Description                                                                                                                                                                                                                                                                 |
|:-----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| datetime   | hourly date + timestamp                                                                                                                                                                                                                                                     |
| season     | 1 = spring, 2 = summer, 3 = fall, 4 = winter                                                                                                                                                                                                                                |
| holiday    | whether the day is considered a holiday                                                                                                                                                                                                                                     |
| workingday | whether the day is neither a weekend nor holiday                                                                                                                                                                                                                            |
| weather    | 1: Clear, Few clouds, Partly cloudy, Partly cloudy 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog |
| temp       | temperature in Celsius                                                                                                                                                                                                                                                      |
| atemp      | "feels like" temperature in Celsius                                                                                                                                                                                                                                         |
| humidity   | relative humidity                                                                                                                                                                                                                                                           |
| windspeed  | wind speed                                                                                                                                                                                                                                                                  |
| casual     | number of non-registered user rentals initiated                                                                                                                                                                                                                             |
| registered | number of registered user rentals initiated                                                                                                                                                                                                                                 |
| count      | number of total rentals                                                                                                                                                                                                                                                     |
## Data Cleaning

- Converted some columns to categorical
- Data imputation on data with `windspeed` value of 0

## Feature Engineering

- Created `week`, `hour`, `day`, `weekday`, `month` columns from `datetime`
- Created `season2` column to replace `season`
- Boxplot analysis on why `season` may not be accurate

## Exploratory Data Analysis

Below are some of the plots during the EDA process:

Outlier Analysis  |  Average Count over Time (Bar)
:-------------------------:|:-------------------------:|                            
![alt text](https://github.com/nchin212/Rental_Bike_Prediction/blob/main/plots/box2.png) |  ![alt text](https://github.com/nchin212/Rental_Bike_Prediction/blob/main/plots/bar1.png) |

Average Count over Time (Line) |  Correlation Analysis
:-------------------------:|:-------------------------:| 
![alt text](https://github.com/nchin212/Rental_Bike_Prediction/blob/main/plots/line1.png) |  ![alt text](https://github.com/nchin212/Rental_Bike_Prediction/blob/main/plots/cor1.png)

## Modelling

Used stepwise regression to create the following models:

**Model 1 -** count ~ workingday + weather + temp + humidity + windspeed + week + hour + day + weekday

**Model 2 -** log(count) ~ workingday + weather + temp + humidity + windspeed + week + hour + day + weekday

## Results

![alt text](https://github.com/nchin212/Rental_Bike_Prediction/blob/main/plots/bar2.png)

Although the log transformation model has a slightly higher RMSE than the linear model, the R squared of the log transformation model is much higher (0.80). This indicates that 80% of the variance of count can be explained by the predictor variables, which makes it the better model.

## Relevant Links

**R Markdown :** https://nchin212.github.io/Rental_Bike_Prediction/rental.html

**Portfolio :** https://nchin212.github.io/post/rental_bike_prediction/
