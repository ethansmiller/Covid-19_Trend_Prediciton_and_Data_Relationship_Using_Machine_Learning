README

Ethan Miller
011075077

Module 6:

BEFORE RUNNING, PLEASE PERFORM 'pip install keras', AND 'pip install tensorflow' IN ANACONDA POWERSHELL PROMPT

Begin by running the module 6 program file.


#1 - Trend Prediction:

For this problem we used a csv file containing global confirmed covid-19 case data. The goal of this problem was to present 3 months of data unaltered, and then using machine learning concepts, predict the behavior of the curve for the next 3 months. In our case, we observed daily confirmed covid-19 data in the US from May to November (6 months total). First we plotted 3 months unaltered from 5/20/2020 to 8/20/2020 (~90 days). This plot can be seen in the 'plots' tab or in the program folder under 'First_3_Months.png'. We then created a model to predict the next 3 months of data based on the first 3 months of data. This plot will be produced in the 'plots' tab or in the program folder as '6_Months_with_Predictions.png'. 
Notes:

Overall our result was reasonable as the prediction trend line begins to follow the expected trend of the original. The trend line mostly strays lower than expected, and this is worth noting as Covid-19 case spikes were much more drastic in real life. If we were to continue observing the data and creating predictions, I believe our model would eventually fail to even remotely match the original trend due to the dynamic factors that lead to the many Covid-19 spikes we experienced in 2020.

*more information about the written code and design of model is specified in comments throughout the code*

Dataset acquired through Kaggle: https://www.kaggle.com/antgoldbloom/covid19-data-from-john-hopkins-university

Code was leveraged from: https://www.youtube.com/watch?v=QIUxPv5PJOY

Assistance from classmates: Matthew, Michael


#3 - Data Relationship

This problem used a dataset for covid cases by county in the state of Maryland. The three criteria we focused on were the counties of Baltimore, Montgomery, and Kent and the recorded cases between them. We first read the csv and placed the data of these three counties into a dataframe. We then created a matrix of plots to observe their relationships, this can be seen in the 'plots' tab or in the program folder under 'MD_relationships.png'. We performed a perason correlation between all three counties, which was printed in the console. Based solely on the Pearson Correlations, we saw that all counties covid cases correlated extremely closely as they were all close to a value of 1. This is understandable as all 3 counties are in proximity to eachother and thus would have very similar rates of confirmed covid cases. This is also confirmed if we looked at the previously mentioned plot, and see that all plots trend linearly. The largest outliers are the relationships between Kent and the two other counties. I noted that this is most likely because Kent is a much smaller county within Maryland, while Baltimore and Montgomery are two of the largest counties in the state. Next we performed a Chi-Square Test for Independence between each county. First, the Chi-Square Test between Baltimore and Montgomery gave us a p-value of 0.239, which shows a strong relationship between the two counties. Repeating for Baltimore and Kent, we got a p-value of 0.271, which also shows a strong relationship. Finally, For Montgomery and Kent we got the same p-value of 0.271. This tells us that Kent has a stronger relationship between the two other counties' covid cases, than Baltimore does with Montgomery. 

Dataset for Maryland Covid Cases by County acquired from Data.gov: https://catalog.data.gov/dataset/md-covid-19-cases-by-county