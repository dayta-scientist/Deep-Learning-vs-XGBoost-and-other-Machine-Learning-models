# Deep Learning vs XGBoost and other Machine Learning models
The main objective of this project is to carry out the development and comparison of the two most used models with the best results obtained in the Kaggle competitions: XGBoost and Deep Learning
We will also present other Machine Learning models, in order to have a measure of the degree of accuracy and adjustment of the dataset to the different applied techniques. In this way, we can see, on the one hand, that the techniques most used in competition have a reason, and on the other, any observer, analyst and decision maker can take a real dimension of the results obtained and thus draw their own conclusions.
For this purpose, the chosen dataset is the scikit-learn Breast Cancer (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)

Deep learning has reached a level of public attention and industry investment never before seen in the history of AI, but it isn’t the first successful form of machine learning.
It’s safe to say that most of the machine learning algorithms used in the industry today aren’t deep learning algorithms. Deep learning isn’t always the right tool for the job—sometimes there isn’t enough data for deep learning to be applicable, and sometimes the problem is better solved by a different algorithm. The only way not to fall into this trap is to be familiar with other approaches and practice them when appropriate. 
Probabilistic modeling is the application of the principles of statistics to data analysis. It is one of the earliest forms of machine learning, and it’s still widely used to this day. One
of the best-known algorithms in this category is the Naive Bayes algorithm.
Kernel methods are a group of classification algorithms, the best known of which is the Support Vector Machine (SVM). But SVMs proved hard to scale to large datasets and didn’t provide good results for perceptual problems such as image classification. Because an SVM is a shallow method, applying an SVM to perceptual problems requires first extracting useful representations manually (a step called feature engineering), which is difficult.
Decision trees are flowchart-like structures that let you classify input data points or predict output values given inputs. It learned from data began to receive significant research interest in the 2000s, and by 2010 they were often preferred to kernel methods.
Random Forest algorithm introduced a robust, practical take on decision-tree learning that involves building a large number of specialized decision trees and then ensembling their outputs.
When the popular machine learning competition website Kaggle (http://kaggle.com) got started in 2010, random forests quickly became a favorite on the platform—until 2014, when gradient boosting machines took over. A gradient boosting machine, much like a random forest, is a machine learning technique based on ensembling weak prediction models, generally decision trees. It uses gradient boosting, a way to improve any machine learning model by iteratively training new models that specialize in addressing the weak points of the previous models. Applied to decision trees, the use of the gradient boosting technique results in models that strictly outperform random forests most of the time, while having similar properties.
From 2016 to 2020, the entire machine learning and data science industry has been dominated by these two approaches: deep learning and gradient boosted trees. Specifically, gradient boosted trees is used for problems where structured data is available, whereas deep learning is used for perceptual problems such as image classification.


Extreme Gradient Boosting: XGBoost
With the acceleration of big data, the search to find awesome machine learning algorithms to produce accurate, optimal predictions began. Decision trees produced machine learning models that were too accurate and failed to generalize well to new data. Ensemble methods proved more effective by combining many decision trees via bagging and boosting. A leading algorithm that emerged from the tree ensemble trajectory was gradient boosting.
The consistency, power, and outstanding results of gradient boosting convinced Tianqi Chen from the University of Washington to enhance its capabilities. He called the new algorithm XGBoost, short for Extreme Gradient Boosting. Chen's new form of gradient boosting included built-in regularization and impressive gains in speed.

After finding initial success in Kaggle competitions, in 2016, Tianqi Chen and Carlos Guestrin authored XGBoost: A Scalable Tree Boosting System to present their algorithm to the larger machine learning community. You can check out the original paper at https://arxiv.org/pdf/1603.02754.pdf. 
The Extreme in Extreme Gradient Boosting means pushing computational limits to the extreme.
Building advanced XGBoost models requires practice, analysis, and experimentation. That way, innovative tips and tricks from the masters at Kaggle, including stacking and advanced feature engineering, will be taken into account.

Stacking proved to be an incredibly powerful method and outperformed the uncorrelated set of models.
Stacking combines machine learning models at two different levels: the base level, whose models make predictions on all the data, and the meta level, which takes the predictions of the base models as input and uses them to generate final predictions. In other words, the final model in stacking does not take the original data as input, but rather takes the predictions of the base machine learning models as input. Stacked models have found huge success in Kaggle competitions. Most Kaggle competitions have merger deadlines, where individuals and teams can join together. These mergers can lead to greater success as teams rather than individuals because competitors can build larger ensembles and stack their models together. Note that stacking is distinct from a standard ensemble on account of the metamodel that combines predictions at the end. Since the meta-model takes predictive values as the input, it's generally advised to use a simple meta-model, such as linear regression for regression and logistic regression for classification.

According to the results obtained after the application of the different machine learning models, it can be verified that stacking was the strongest result: 98.068%.


Deep Learning:
The necessary steps were carried out to obtain the results in terms of the highest degree of precision possible. Considering that the dataset is pre-processed by scikit-learn, there are steps that are not necessary to perform. So, the first step was to standardize the data. Then, a create_model () function was defined that creates a multilayer neural network for the problem. We pass this function name to the KerasClassifier class by the build_fn argument. The options are specified into a dictionary and passed to the configuration of the GridSearchCV. The performance and the combination of configurations for the best model are obtained.
Finally, the best combination is applied and the accuracy of the train set is 99.50% and that of the test set is 98.25%.


Conclusion:

The best result that could be obtained was with the Deep Learning model, surpassing XGBoost, at the same time as predicted by the theory, surpassing the other Machine Learning models.

