WARNING: main_income takes a long time to run due to how large the dataset is for SVM. If you want to speed things up, I suggest using a subset of the training/test set (pd.read_csv('./data/adult.data').iloc[:500])

This assignment was run on python 3.7.4.
Virtual Env: Anaconda 
Environment: Windows 10
IDE: Spyder

#For other person to use the environment
conda env create -f <environment-name>.yml

I have attached 2 files for running my code:
	ml.yml
	requirements.txt

ml.yml is used to create a similar anaconda environment as the one I have.
requirements.txt lists all the libraries used and their corresponding versions.

The main code for my datasets are:
	main_income.py (main function for the income dataset)
	main_admission.py (main function for the graduate admissions dataset)

Graduate Admissions Dataset: https://www.kaggle.com/mohansacharya/graduate-admissions
Income Dataset: http://archive.ics.uci.edu/ml/datasets/Adult

There are 2 subfolders:
	/data/
	/modules/

The data folder contains the datasets graduate_admissions and the test/training dataset for income.
The modules folder contains the necessary modules for modeling my data and plotting the results.