# project-himuna-data4good

### Recent Updates

+ ***12/03/2023:***
   + We are  presently engaged in updating a report for Humana Mays, incorporating changes since its initial due date. Additionally, I am concurrently working on a report for Data 4 Good.
+ ***11/27/2023:***
   + Currently in the process of writing detailed reports to document our approach, methodologies, and findings for both Humana competitions according to the given format with 80% complete.
  
+ ***11/13/2023:*** 
   + Leaderboard Achievement: Secured 46th position on the leaderboard, showcasing our team's competitive performance.

   + Successful Execution and Submission: Accomplished the successful execution of our initial run, demonstrating the robustness of our model. Submitted our model in a timely manner, aligning with project milestones and deadlines.
+ ***11/06/2023:*** 
  + Token issues: The LLM hosting service Prediction Guard has experienced some issues, which are still being addressed..
  + We have submitted our first basic model on kaggle. Now we are working on to improve the model and score.


+ ***10/29/2023:*** 
  + **Output Processing**: We were unable to produce an output as platform (Pediction Guard) hosting these LLMs have been going through their own issues which should be solved by tomorrow as per their email.
  + **Data Input**: We have provided the LLM with input data. We have added the structure for the prompt.
  + We have started working on the modelling LLM for Data4good competition. It includes studying and reading more on the Large Language Model (LLM) and its API.

  



### ***Work Updates So Far:***

> 1. Data Preprocessing (target):

   + Missing values in the target_train dataset are handled by filling them with appropriate values (e.g., mode, median) for columns 

   + The  columns are converted to date-time objects and made timezone-aware. The variables are calculated in days.

   + One-hot encoding is applied to the one column to convert it into numerical features.

  > 2. Data Preprocessing (Train):


+ Missing values in certain columns of are filled 

+ Date columns are converted to datetime objects.

+ Specific columns have their missing values filled with medians.


> 3. Feature Engineering and Merging:

+ Aggregated features are created for the train datasets, such as the number of unique diagnoses, total ADE diagnoses, total prescription costs, and total drug-drug interaction indications. These features are aggregated.
  
+ These aggregated features are merged into the target dataset.

> 4. Handling Class Imbalance:

+ The SMOTE algorithm is used to address class imbalance by oversampling the minority class in the training data (X_train_resampled and y_train_resampled).
  
> 5. Model Training and Evaluation:


+ Four classification models are trained and evaluated: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.


+ The model is trained on the resampled training data and predictions are made on the test data.
 
+ ROC AUC score and a classification report (precision, recall, F1-score, etc.) are calculated and printed.


### ***Previous Updates***
+ ***10/22/2023:*** We are making progress at a slow pace in our exploration of data for the Data4Good competition, as we are currently juggling multiple commitments simultaneously.
  + Initial Data exploration: Explored the raw data to gain a preliminary understanding of its structure, format, and potential challenges.
  + Understanding of the problem and the approach: Gained a deep understanding of the problem or task posed by the competition. Considering the specific goals and objectives.  
  + We have started working on the Data4Good case competition. Got the data. 
+ ***10/15/2023:*** We have completed and turned in the Round-2 report at 11 PM and are now looking forward to the results with anticipation.
+ ***10/07/2023 - 10/15/2023:*** We achieved a commendable **19th place** ranking among the top 50 qualifiers and we are working on the report.
  + Report Work Split:
     + **Anirudh and Tharun** - Responsible for report writing. They will likely be in charge of creating the content of the report, organizing it logically, and ensuring that it flows well.
     + **Abdul** - In charge of Descriptive Analytics for the report. This may include analyzing data, creating visualizations, and providing insights to support the content of the report.
     + **Praful** - Responsible for proofreading and maintaining the format, grammar, and style of the report. This is an essential role to ensure that the final document is polished and error-free.

  
+ ***10/06/2023:*** We have successfully completed the necessary revisions and have subsequently submitted the final version for round-1.
  
+ ***10/05/2023:*** Reworked Data Engineering and updated variables to input for modeling. We improved our modeling results to look much better. Also, Our position dropped from 41 to 46.
+ ***10/06/2023:*** We have successfully completed the necessary revisions and have subsequently submitted the final version for round-1.
  
+ ***10/05/2023:*** Reworked Data Engineering and updated variables to input for modeling. We improved our modeling results to look much better. Also, Our position dropped from 41 to 46.

+ ***10/04/2023:*** Reworked Data Engineering and updated variables to input for modeling. Also, Our rating dropped to position 41.
  
+ ***10/03/2023:*** There is no update since our teammates are preoccupied with other commitments.
 
+ ***10/02/2023:*** There is no update since our teammates are preoccupied with other commitments. 

+ ***10/01/2023:*** There is no update since our teammates are preoccupied with other commitments. 
  
+ ***09/30/2023:*** Following the current submission for Humana, our rating dropped from the 21st to the 24th. Unfortunately, it looks like our most recent submittal produced lower accuracy than the prior one. We are working on  improving the accuracy 
+ ***09/29/2023:*** Our ranking slipped from 14th to 21st after the recent submission for Humana. Unfortunately, it appears that our latest submission yielded lower accuracy compared to the previous one. However, we are actively working on improving our AUC score, and we are considering revisiting the data engineering aspect to optimize data performance, which could potentially help us regain our previous standing.

+ ***09/28/2023:*** We've successfully addressed the issues in our submission file, and as a result, we've climbed to the **17th position** on the leaderboard. Our ongoing efforts are focused on refining our models to further enhance our AUC score.

+ ***09/27/2023:*** Our file name, presented to the leadership board yesterday, had a mistake. We ran the second model with a higher AUC than the first and submitted it to the leadership board with the proper file name.

+ ***09/26/2023:*** Completed the data engineering portion, completed the first model, and completed the first submission to Humana. Further, we will be making more models to get a higher AUC score.

+ ***09/25/2023:*** We finished the data engineering section and decided what models we could create by using Lazy Predict in Python to obtain an idea of what models we could build.

+ ***09/24/2023:*** Our initial phase involved conducting a comprehensive descriptive analysis of the variables within our dataset. This encompassed an examination of the variable distributions to identify any instances of missing data, along with a thorough assessment to detect potential outliers.

Also, we engaged in a detailed discussion about the crucial area of data engineering. This encompassed two pivotal aspects

Data Transformation: We contemplated various methods for refining and preparing the data to ensure its suitability for further analysis.

Data Merging: We explored strategies for consolidating multiple datasets to create a unified and comprehensive dataset for our analysis.
 

+ ***09/22/2023***: Exploring the data and doing descriptive analytics on variable distribution and outliers, as well as designing the portion for data Engineering to integrate data 

+ ***09/18/2023***: We haven't received the data 



