# AC 221 Final Project
Abhimanyu Vasishth and Phoebe Wong

## Evaluating Algorithmic Fairness on the NYPD Stop, Question and Frisk Dataset

The aim of this project is to study different definitions of fairness on a dataset about human decision making that lends itself to implicit biases (such as biases pertaining to race or gender) or to models that may violate one or many definitions of fairness. To this end, we use the Stop, Question and Frisk dataset from the New York Police Department (NYPD). Given that numerous police departments are already utilizing algorithms for predictive policing, it is not unreasonable to imagine the NYPD leveraging algorithmic models to identify the risk factor for suspects they stop in determining whether or not to arrest them. Specifically, we aim to build a model that has a high classification accuracy on whether a person who was stopped, questioned and frisked was arrested or not and then evaluate this model using notions of fairness such as statistical parity and conditional parity, and also evaluate the decision making process of the model based on feature importances and discuss the applicability of important features in a real-world setting, i.e. if the NYPD deployed an algorithmic solution towards assessing the risk percentages of suspects stopped and using these risk scores to determine whether a suspect should be arrested or not. Furthermore, we aim to provide local interpretability through SHAP values and understand potential solutions to dealing with biased data.

## Deliverables

View the full report [here](https://github.com/phoebewong/NYPD-Algorithm-Fairness/blob/master/src/Report_submit.ipynb). 

