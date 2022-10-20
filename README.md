# Haiti-Disaster-Relief-Project

## Introduction
Following the 2010 Haiti earthquake, rescue workers needed to get food and water to the displaced persons. But with destroyed communications, impassable roads, and thousands of square miles, actually locating the people who needed help was challenging.

As part of the rescue effort, a team from the Rochester Institute of Technology were flying an aircraft to collect high resolution geo-referenced imagery. It was known that the people whose homes had been destroyed by the earthquake were creating temporary shelters using blue tarps, and these blue tarps would be good indicators of where the displaced persons were – if only they could be located in time, out of the thousands of images that would be collected every day. The problem was that there was no way for aid workers to search the thousands of images in time to find the tarps and communicate the locations back to the rescue workers on the ground in time. The solution would be provided by data-mining algorithms, which could search the images far faster and more thoroughly (and accurately) then humanly possible. The goal was to find an algorithm that could effectively search the images in order to locate displaced persons and communicate those locations rescue workers so they could help those who needed it in time.

<img src="https://github.com/HaleyEgan/Haiti-Disaster-Relief-Project/blob/main/exampleimage.png" width="400"/>

## Project Details
The goal of this disaster relief project is to test each of the algorithms on the imagery data collected during the relief efforts made in Haiti in response to the 2010 earthquake, and determine which method to use, to as accurately as possible, and in as timely a manner as possible, locate as many of the displaced persons identified in the imagery data so that they can be provided food and water before their situations become un-survivable.

The project documents the performance of Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), K-Nearest Neighbor (KNN), Penalized Logistic Regression, Random Forest, and Support Vector Machines (SVM) using 10-fold cross-validation and a hold-out dataset. The data set is from the 'HaitiPixels.csv' file, which contains three columns, 'Class', 'Red', 'Green', and 'Blue'. The 'Red', 'Green', and 'Blue' variables are the pixel data collected at each location. The 'Class' variable are categories of objects found in the images per location, classified as 'vegetation', 'soil', 'rooftop', 'various non-tarp', and 'blue tarp'. For this project, the response/outcome variable is 'blue tarp' from the 'Class' column. 'Red', 'Green', and 'Blue' are used as the predictor variables. This data is used to test the five different algorithms, to determine which is the best to predict the location of blue tarps.

### Cross-Validation Performance Table

| Model |	Tuning | AUROC | Threshold | Accuracy | TPR | FPR |  
| ----- | ------ | ----- | --------- | -------- | --- | --- | 
| Log Reg |	NA	| 0.9985696 | 0.05 | 0.9936355 | 0.9996937 | 0.1892827 | 
| LDA	|	NA |	0.9892327 |	0.05 | 0.9845512 | 0.9935968 | 0.2892894 |  
| QDA	|	NA	| 0.9982114 | 0.05 | 0.9920345 | 1 | 0.2484322 | 
| KNN	|	k=9	|	0.9998489 | 0.05 | 0.992568 | 0.999755 | 0.2275 | 
| Pen Log Reg |lasso, a=1, l=0.001 | 0.9975849 | 0.05 | 0.9837526 | 1 | 0.5068091 | 
| RF | m=6, ntree=100 | 0.9947216 | 0.05 | 0.9949202 | 0.9998366 | 0.1535409 | 
| SVM | radial kernel, c=10 | 0.9996892 | 0.05 | 0.994347 | 0.9996324	| 0.1652352 |

| Model |  Precision | FNR | 
| ----- | ------ | --------- | 
| Log Reg | 0.9937687 | 0.00030631 |
| LDA | 0.9953286 | 0.0149627 | 	
| QDA	| 0.9918392 | 0 |
| KNN	| 0.9926238 | 0.0002449647 |  	
| Pen Log Reg | 0.9834930 | 0 |
| RF | 0.9949414 | 0.0001633653 |
| SVM | 0.9945568 | 0.0003675678 |

### Hold-Out Data Performance Table

| Model |	Tuning | AUROC | Threshold | Accuracy | TPR | FPR |   
| ----- | ------ | ----- | --------- | -------- | --- | --- |  
| Log Reg |	NA	| 0.9996137 | 0.05 | 0.9936355 | 0.9996937 | 0.1892827 | 
| LDA	|	NA | 0.9930592 |	0.05 | 0.9845512 | 0.9935968 | 0.2892894 | 
| QDA	|	NA	| 0.7870076 | 0.05 | 0.9920345 | 1 | 0.2484322 | 
| KNN	|	k=9	|	0.9618915 | 0.05 | 0.992568 | 0.999755 | 0.2275 | 
| Pen Log Reg | lasso, a=1, l=0.001 | 0.9994029 |0.05 | 0.9837526 | 1 | 0.5068091 |  
| RF | m=6, ntree=100 | 0.9689964 | 0.05 | 0.9949202 | 0.9998366 | 0.1535409 | 
| SVM | rad kernel, c=10 | 0.9908996 | 0.05 | 0.9942087 | 0.9991832 | 0.1559721 |

| Model |  Precision | FNR | 
| ----- | ------ | --------- | 
| Log Reg | 0.9937687 | 0.00030631 |
| LDA | 0.9953286 | 0.0149627 | 	
| QDA	| 0.9918392 | 0 |
| KNN	| 0.9926238 | 0.0002449647 |  	
| Pen Log Reg | 0.9834930 | 0 |
| RF | 0.9949414 | 0.0001633653 |
| SVM | 0.9945568 | 0.0003675678 |

*The metrics were calculated by using averages. 

## Conclusions 
A thorough analysis of seven supervised learning algorithms (models) was conducted to determine which would perform best on the imagery data collected during the relief efforts made in Haiti in response to the 2010 earthquake. The goal was to determine which method to use to locate as many of the displaced persons as possible to provided food and water. 'Blue Tarps' were used as the response/outcome variable, and 'Red', 'Green', and 'Blue' pixel data were used as the predictors of blue tarps. Analysis was performed on Logistic Regression, Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), K-Nearest Neighbor (KNN), Penalized Logistic Regression, Random Forest (RF), and Support Vector Machine (SVM) models using 10-fold cross-validation. A holdout data set of over 2 million data points was used to further test the performance of each model.

### Algorithm Performance, Analysis & Recommendation
For each model, ROC and AUC were examined, as well as statistical metrics for threshold, accuracy, true positive rate (TPR), false positive rate (FPR) and precision. Upon comparison of the seven different models, it was clear that all models performed very well and could be useful with locating blue tarps, and thus persons in need of aid in Haiti. It was determined that the best approach for the statistical threshold metrics was selecting the metrics with the smallest false negative rate (FNR). With a lower false negative rate, less people are likely to be missed due to error. For example, it would be better to wrongly identify something that is actually ‘soil’ as ‘blue tarp’, than to wrongly classify a ‘blue tarp’ as ‘soil’. In other words, it would be most harmful to miss people that are in need, than to call ‘soil’ a blue tarp. Therefore, in examining the threshold output, the False Negative Rate is used. False Negative Rate favors a low threshold, so 0.05 was used. 

By examining the metrics for each model based on false negative rate, the best performing models using the training data were QDA, SVM, and Random Forest. All three models have high levels of accuracy, precision, true positive rates, and false negative rates. They also have well performing ROC and AUC curves when using the training data. When using the holdout data set, the best performing models are Penalized Logistic Regression, Logistic Regression, LDA, and SVM. While QDA performs very well on the training data, it is the worst performing model on the holdout data. 

When looking at both the training data set and holdout data set, this leaves the SVM radial kernel as the all around best performing model. The SVM algorithm has the highest chance of identifying blue tarps, the lowest change of missing blue tarps, and thus the highest likelihood of success and efficiency in the Haiti search mission. However, Support Vector Machines require a high level of computational power and time when analyzing large data sets. This is a potential problem if results are needed quickly, and if there are computational constraints. Logistic Regression could be a good choice to achieve both highly accurate and efficient results. 

Since this project is in the effort of solving a time-sensitive humanitarian crisis, choosing a time and computationally efficient, yet highly effective algorithm is ideal. Therefore, using the Logistic Regression model is recommended to meet all needs. It performs well, and is rapid, even when using large amounts of data points. Logistic Regression performs very well for both the training data and the holdout data. 

### Relevance of Metrics
The metrics that are being measured for each model are important in communicating the abilities and accuracy of the models. The AUROC curve is a good way to visualize accuracy of the model, and the statistical threshold metrics allow for quick comparisons of each model. Accuracy, True Positive Rate, Precision, and False Negative Rate are the most important metrics to measure for this specific project, and in choosing the best performing model.

In order to further improve results for the model selection, it is recommended to run an analysis on each model to determine its processing speed, and adding these metrics to the performance tables. When calculating lots of data, these models may not perform as well as they are now, due to the calculations that are needed for the model. If some models are significantly slower than others, they would not be recommended for use, due to the urgency of this search mission. If a model is successful in all metrics, but is too slow like Support Vector Machines, then a different model may be preferred. If the metrics show that there is not a large difference between time 

### Using Blue Tarps
The analysis on the models above could be effective in terms of helping save a human life. However, there are many other data-oriented factors that should be considered in a real scenario. For example, in this case only blue tarps are looked at. While this can be an important place to start for on-the-ground support crews, there may be many other places with people in need that are not constrained to blue tarps. For example, shelters may be made from vegetation. Additionally, the data set contains a classification for 'roof tops'. This could be an important class to add to our analysis and on-the-ground search, since many people may have stayed at their homes after the disaster, and also do not have the necessary food and water. It may also be worth knowing how many people on rural Haiti are likely to own blue tarps before the disaster. While that may be a prevalent tool to use for shelter after a disaster in some countries or regions, this may not be universal. Before beginning an algorithm analysis, it may be important to consider these, and possibly more, factors. If this is done, a more thorough rescue mission may be possible, and varying factors can be incorporated into the model analysis. It is also recommended to use a larger training and test set, since in this case, the category of "True" blue tarps was limited in size. This could also change the outcome of the models.

### Application
This project can be applied to other types of "blue tarp" projects around the world. If there are other natural disasters where people need to be located to provide aid, a similar approach can be taken. A similar application could be identifying the location and quantity of houseless persons in the United States through imagery identification. This type of project can also be used as an example for other projects spanning a wide variety of topics. It is often necessary to test multiple models in order to choose the most effective and efficient model for the particular data set. The type of model/algorithm that is best for the data will change depending on the data and the use case. It is important to perform exploratory data analysis to examine the type of data and their relationships, and then to test multiple models on training and test data sets. The structure of this project can be applied to most other classification data sets, in order to determine the best supervised learning algorithm. 

*Since this is an ongoing project for students, code will be provided upon request. 
