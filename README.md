# Kaggle
# This is a in Class Toxicity Prediction Challenge 
# Programming Language: Python
# It requires sklearn, numpy, pandas and light gbm installation along with Python on your system
Following libraries can be installed directly using Operating System: MAC OS 11.2
RUN pip install lightgbm
RUN pip install pandas
RUN pip install -U scikit-learn
RUN pip install numpy

# Instructions to run the code:

Prerequisites:
• Install - Jupyter Notebook (anaconda3).
• Download train, test, and feamat datasets from “The Toxicity Prediction Challenge”
competition (Kaggle) or from this GIT repository

#Execution:

1. Launch the Jupyter Notebook.
2. Create a folder with the name “ToxcityPrediction” in Jupyter Notebook.
3. Upload the feamat, train, test datasets and FinalCode.py to the folder ToxcityPrediction 
4. open terminal and go to the folder ToxcityPrediction and run the command python FinalCode.py
5. In the terminal FinalCode.py will get executed and whole execution would take about an hour.
6. Once the execution completes, submission_final_lgb.csv file will be created in the same folder where FinalCode.py was executed i.e.,ToxcityPrediction folder


NOTE - Specify the path of your own directory for read.csv files where feamat, train, test datasets are kept.

For Example - If feamat, train, test datasets are kept in ToxcityPrediction folder then the following line of code would be replaced by - 

#load train dataset
train = pd.read_csv('/ToxcityPrediction/train.csv')   

#load test dataset
test = pd.read_csv('/ToxcityPrediction/test.csv') 

#load feamat dataset 
feature_matrix = pd.read_csv('/ToxcityPrediction/feamat.csv')  





