# DataBots_Kaggle
# This is a Kagglel InClass Toxicity Prediction Challenge 
# Programming Language: Python

 It requires sklearn, numpy, pandas and light gbm installation along with Python on your system
 
 Operating System: MAC OS 11.2

 Following libraries can be installed directly 
 
1) RUN pip install lightgbm;

2) RUN pip install pandas;

3) RUN pip install -U scikit-learn;

4) RUN pip install numpy;

# Instructions to run the code:

Prerequisites:
• Install - anaconda3 or python.
• Download train, test, and feamat datasets from “The Toxicity Prediction Challenge”
competition (Kaggle) or from this GIT repository

#Execution:


1. Create a folder with the name “ToxcityPrediction” 
2. Upload the feamat, train, test datasets and FinalCode.py to the folder ToxcityPrediction 
3. open terminal and go to the folder ToxcityPrediction and run the command python FinalCode.py
4. In the terminal FinalCode.py will get executed and whole execution would take about an hour.
5. Once the execution completes, submission_final_lgb.csv file will be created in the same folder where FinalCode.py was executed i.e.,ToxcityPrediction folder


NOTE - Specify the path of your own directory for read.csv files where feamat, train, test datasets are kept.

For Example - If feamat, train, test datasets are kept in ToxcityPrediction folder then the following line of code would be replaced by - 

#load train dataset
train = pd.read_csv('/ToxcityPrediction/train.csv')   

#load test dataset
test = pd.read_csv('/ToxcityPrediction/test.csv') 

#load feamat dataset 
feature_matrix = pd.read_csv('/ToxcityPrediction/feamat.csv')  





