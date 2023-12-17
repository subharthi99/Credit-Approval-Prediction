With this project we aim to compare supervised learning (SL) and semi-supervised (SSL) machine learning techniques and analyze and interpret their performances using the Credit Approval Dataset from the Keel Machine Learning Dataset - Resembling a case study from Capital One.

![Capital One Logo](assets/capital-one.png)

For the SL case, we have identified a trivial classifier which randomly outputs class probabilities weighted by training priors, two baseline classifiers (Nearest Centroid and Logistic Regression) and five machine learning models (CART Decision Tree, AdaBoost, KNN Classifier, Random Forest and SVM Classifier). We used KNN Imputer to handle missing data, RFE, Chi-Square and GridSearch techniques for Feature Selection, PCA for dimensionality reduction, GridSearchCV for parameter selection and trained each model. We then used the best model to evaluate the test-set performance.

For the SSL case, we have used three categories of datasets - 30% labeled data, 20% labeled data and 10% labeled data. Then, we identified four baseline classifiers and four models - Self-Training Classifier with Logistic Regression and SVC (with baselines Logistic Regression and SVC on 100% labeled data), Semi-Supervised SVM Classifier - 2 approaches (with baseline SVC on 100% labeled data), Semi-Supervised Gaussian mixture model - 2 approaches (with baseline GMM on 100% labeled data). We compare the results of these models with the baseline models to gauge performance. We did not use any feature selection or parameter selection techniques, as the goal here is to understand how unlabeled data affects model performance.

In both SL and SSL cases, we compare performances using three metrics, accuracy, macro f1 score and confusion matrix on the test set. For SL, We observe that the Random Forest Classifier with GridSearchCV (best parameters: 'criterion': ‘entropy’, 'max_depth': 5, 'n_estimators': 100, 'max_features': ‘auto’) had the best accuracy of 0.8995. For SSL, the Self-Training Classifier with Logistic Regression has the best accuracy for 30% labeled data (0.803) and 20% labeled data (0.758), and the GMM - Approach 2 has the best accuracy for 10% labeled data (0.742).



Files to run:
EE660_Project_SemiSupervised.ipynb
EE660_Project_SupervisedLearning.ipynb

Required modules that need to be installed:
pip install mlxtend
pip install pandas
pip install numpy
pip install matplotlib
pip install sklearn
pip install scipy

Additional instructions to set appropriate directory:
If running on colab:
1) 	Upload entire Submissions folder to Drive
2) 	Change the following line of code:
	os.chdir("/content/gdrive/Shared drives/EE660_Proj/Submissions")
	to match the location of Submissions. 
	For eg, if the Submissions folders resides in root directory of Google drive, the path will be:
	os.chdir("/content/gdrive/MyDrive/Submissions")
3) 	Run all cells (allow access to Drive when the prompt appears)

If running on Jupyter notebook:
1) 	Upload entire Submissions folder to colab
2) 	Delete cell 1, where we mount drive.
	Lines of code deleted:
	from google.colab import drive
	drive.mount("/content/gdrive", force_remount=True)
	from google.colab.data_table import DataTable
	DataTable.max_columns = 60
3) 	Change the following line of code:
	os.chdir("/content/gdrive/Shared drives/EE660_Proj/Submissions")
	to match the location of Submissions. 
	For eg, if the Submissions folders resides in 'OneDrive/Desktop' for User 'Test' (Windows), the path will be:
	os.chdir("C:/Users/Test/OneDrive/Desktop/Submissions")
4) 	Run all cells

Note: If at any point the code execution halts due to Warnings, just re-run all cells from the cell containing all imports. Do not restart runtime or kernel.

Additional comments:
1) 	In EE660_Project_SupervisedLearning.ipynb all implementations of Random Forest Classifier take approximately 15 minutes to run.
2) 	Please make sure to run the ipynb files inside the Submissions folder (It contains a .py file that is being imported in EE660_Project_SemiSupervised.ipynb). If not, this file will have to be manually added to the current path.
