# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 15:49:44 2015
@author: NDT567
"""
# -*- coding: utf-8 -*-

import numpy
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
from ggplot import *
from sklearn import metrics
from sklearn import tree
import csv
import os
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb
import sklearn.cross_validation as cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model, datasets
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn import cluster, datasets
from IPython.display import Image  
from sklearn.externals.six import StringIO  
dot_data = StringIO()  
from sklearn import metrics
from sklearn import tree

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.learning_curve import validation_curve
from sklearn.grid_search import GridSearchCV   #Perforing grid search

"""
This is the module to clean the dataset and covert all categorical variables to binary columns
df is the build database and df_val is the validation database

The module will make sure the build db and the validation db have the same data structure.
"""
def middle_out(df,df_val):
 
# create a list of unique categorical values from all categorical columns in the build db
    build={}
    for i in range (0,len(df.select_dtypes(include=['object']).columns.values)):
        build[df.select_dtypes(include=['object']).columns[i]]=[pd.unique(df.select_dtypes(include=['object']).ix[:,i])]

# create a list of unique categorical values from all categorical columns in the validation db        
    val={}
    for i in range (0,len(df_val.select_dtypes(include=['object']).columns.values)):
        val[df_val.select_dtypes(include=['object']).columns[i]]=[pd.unique(df_val.select_dtypes(include=['object']).ix[:,i])]

# create an exception list of categorical values
    diff_val={}
    for i in build.keys():
        diff_val[i]=set(numpy.array(build[i]).flat)-set(numpy.array(val[i]).flat)
    diff_build={}
    for i in build.keys():
        diff_build[i]=set(numpy.array(val[i]).flat)-set(numpy.array(build[i]).flat)
    
# insert the new categorical values into the origianl database
    diff_val_df = pd.DataFrame.from_dict(diff_val, orient='index').transpose()
    diff_build_df = pd.DataFrame.from_dict(diff_build, orient='index').transpose()
    result_val = pd.concat([df_val, diff_val_df], ignore_index=True)
    result_build = pd.concat([df, diff_build_df], ignore_index=True)

# convert all categorical columns into binary columns   
    d00=pd.get_dummies(result_build,dummy_na=True)
    d01= pd.get_dummies(result_val,dummy_na=True)

# drop the dummy rows 
    if len(diff_val_df)>0:
        d01=d01.drop(d01.tail(len(diff_val_df)).index)

    if len(diff_build_df)>0:
        d00=d00.drop(d00.tail(len(diff_build_df)).index)
    return d00, d01

"""
This is the module that down size the data
input is dataframe, target variable, and desired non event and event distribution
"""



def down_size (df, target, non_evt, evt):
    if len(pd.unique(df[target]))==2:      
        if len(df.loc[df[target] == pd.unique(df[target])[0]])>len(df.loc[df[target] == pd.unique(df[target])[1]]):
            df_down=df.loc[df[target] == pd.unique(df[target])[0]]
            df_oth=df.loc[df[target] == pd.unique(df[target])[1]]
            cut_off=float(non_evt)/evt*len(df_oth)/len(df_down)
        else:
            df_down=df.loc[df[target] == pd.unique(df[target])[1]]
            df_oth=df.loc[df[target] == pd.unique(df[target])[0]]
            cut_off=float(non_evt)/evt*len(df_oth)/len(df_down)
            
        return pd.concat([df_oth, df_down.loc[numpy.random.uniform(0,1,len(df_down)) <=cut_off]], ignore_index=True)
    else:
        print "The target variable does not contain 2 unique values"


"""
This is the module that plot quantile chart
"""


def quantile_plot(data, var, cut, y):
    
    data['ranks']=pd.qcut(data[var],cut,labels=False)
    grpby=data.groupby('ranks').mean()

    plt.plot(grpby[y])
    plt.ylabel('avg ' + y)
    plt.show()


"""
This is the module that export the confusion matrix to an excel sheet.
The continuous confusion matrix was stored in a dictionary with multiple entries

zz is the CCM, b is the destination file you specified.
"""

def export_CCM (zz, b):

    workbook = xlsxwriter.Workbook(b )
    worksheet = workbook.add_worksheet()
    row = 1
    col = 0
    for k in zz:
        v=zz[k]
        worksheet.write(row, 4, k)
        for i in range (0,2):
            for dd in range (0,2):
                worksheet.write(row, col, v[i,dd])
                col +=1
        col = 0
        row += 1
    workbook.close()


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:18:20 2015

@author: TRY137
"""
#Function that prints out all the top X features of a predicted model 
#predicted_fit = your classifier 
#training_var = your training dataset without the target variable (eg. d0)
    
    
def features_print(predicted_fit, training_var):
    importances = predicted_fit.feature_importances_
    indices = numpy.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(len(training_var.columns.values)):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]), training_var.columns.values[indices[f]])



def features_print2(predicted_fit, training_var):
   indices = numpy.argsort(predicted_fit.feature_importances_)
   # plot as bar chart
   plt.barh(numpy.arange(len(training_var.columns)), predicted_fit.feature_importances_[indices])
   plt.yticks(numpy.arange(len(training_var.columns)) + 0.25, numpy.array(training_var.columns)[indices])
   _ = plt.xlabel('Relative importance')


#test goes first, then predicted 
#prints out ROC curve as well as 

def roc_plot (test_target, predicted_target):

    fpr, tpr, _ = metrics.roc_curve(test_target, predicted_target)    
    auc = metrics.auc(fpr,tpr)
    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    print (ggplot(df, aes(x='fpr', y='tpr')) +\
        geom_line() +\
        geom_abline(linetype='dashed'))
    print 'AUC for your ROC curve:'+str(auc)

# module for printing out the CART tree rule set above certain threshold
#d0 is the dataset for attribute, d1 is the target variable values, 
#level is depth of tree, min_leaf is minimal saples required 
#cut off is the threshold


def print_cart (d0, d1, level, min_leaf, cutoff):
    
    # train the CART model
    cart=tree.DecisionTreeClassifier(random_state=0,min_samples_leaf=int('%d' % (min_leaf)), max_depth=int('%d' % (level)))
    cart=cart.fit(d0, d1)
    #set up all relavant arrays with the branches, threshold, features and values
    left      = cart.tree_.children_left
    right     = cart.tree_.children_right
    threshold = cart.tree_.threshold
    features  = [d0.columns[i] for i in cart.tree_.feature]
    value = cart.tree_.value
    
    # pick all indexes for all the terminal nodes, that is left or right with -1
    indexes = [i for i,x in enumerate(left) if x == -1]
    
    d=0
    # start of the while loop
    while d<len(indexes):
        # set up the threshold 
        if float(value[indexes[d]][0][1]/(value[indexes[d]][0][1]+value[indexes[d]][0][0])) >= float('%f' % (cutoff)):
            #starting with each terminal node
            aaa=indexes[d]
            print "level %s , leaf node" % (level) + str(indexes[d]) + " and set:" + str(d)
            while aaa>0:
                #if the index is not in the left branch
                if [i for i,x in enumerate(left) if x == aaa]==[]:     
                    #print the  condition with the > sign
                    print  " " + str(features[right.tolist().index(aaa)]) + " > " + str(threshold[right.tolist().index(aaa)])
                    aaa=right.tolist().index(aaa)
                    #if it is the terminal node, print the values and fraud rate
                    if aaa==0:
                        print str(value[indexes[d]])
                        a=value[indexes[d]][0][1]/(value[indexes[d]][0][1]+value[indexes[d]][0][0])*100
                        print ("%.2f" % a) + "%"
                    else:
                        print " and "
                else:
                    #if the condition is in the left branch, print the condition with <= sign
                    print  " " + str(features[left.tolist().index(aaa)]) + " <= " + str(threshold[left.tolist().index(aaa)]) 
                    aaa=left.tolist().index(aaa)
                    if aaa==0:
                        print str(value[indexes[d]])
                        a=value[indexes[d]][0][1]/(value[indexes[d]][0][1]+value[indexes[d]][0][0])*100
                        print ("%.2f" % a) + "%"
                    else:
                        print " and "
        d=d+1    




#make a function to plot univariate charts for all variables in a dataframe
def plot_univariate (df=df):
    #plot categorical variables using bar chart
    for i in df.select_dtypes(include=['object']).columns:
        df[i].value_counts().plot(kind='bar', title=i)
        plt.show()
    #plot continuous variables with histogram
    for i in df.select_dtypes(exclude=['object']).columns:
        df[i].hist()
        plt.title(i)
        plt.show()
    #plot box plot for all continuous variables
    for i in df.select_dtypes(exclude=['object']).columns:
        sb.boxplot(df[i])
        plt.show()






#This module will plot bivariate analysis
def bi_variate(df,var1,target):
    rate=df.groupby([var1])[target].sum()/df.groupby([var1])[target].count().sort_index()
    vol=df[var1].value_counts().sort_index()
    pl=pd.concat([vol, rate],axis=1)
    print (pl)
    fig = plt.figure()
    ax = pl[var1].plot(kind="bar");
    ax.set_xlabel(var1)
    plt.xticks(rotation=90)
    ax2 = ax.twinx()
    ax2.plot(ax.get_xticks(),pl[target],marker='o',color='red')
    plt.title(target + ' by ' + var1)
    ax2.set_ylim(ymin=0)
    plt.show()



#cross_validation module, enter 0 for 1 fold train, test split, enter 1 for k-fold algorithm and 2 for stratified k-fold
def cross_validation_df (df, kf_type, test_size=0, stratify=None, n_folds=None, shuffle=False, random_state=None):
    final_val=[]
    if kf_type==0:
        print ("1 fold train test split, "+ str(test_size*100) + "% test sample is used")
        X_train, X_test = cross_validation.train_test_split(df, test_size=test_size,stratify=stratify, random_state=random_state)
    if kf_type==1:
        print (str(n_folds) +"-fold algorithsm is used")
        kf = KFold(len(df),n_folds=n_folds,shuffle=shuffle, random_state=random_state)
        i=0
        X_train, x_test=[],[]
        for train_index, test_index in kf:
            print("TRAIN:", train_index, "TEST:", test_index)
            vars()["X_train"+str(i)],vars()[ "X_test"+str(i)] = df.ix[train_index], df.ix[test_index]
            final_pd=final_pd.append(vars()[ "X_test"+str(i)], ignore_index=True)
            i+=1
    if kf_type==2:
        print (str(n_folds) +"-stratified fold algorithsm is used")
        skf = StratifiedKFold(stratify.values, n_folds=n_folds, shuffle=shuffle,random_state=random_state)
        i=0
        X_train, x_test=[],[]
        for train_index, test_index in skf:
            print("TRAIN:", train_index, "TEST:", test_index)
            vars()["X_train"+str(i)],vars()[ "X_test"+str(i)] = df.ix[train_index], df.ix[test_index]
            i+=1


#module to plot variable importance from different algorithsm
def features_print(predicted_fit, training_var):
    importances = predicted_fit.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("Feature ranking:")
    for f in range(len(training_var.columns.values)-1):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]), training_var.columns.values[indices[f]])
        
#module to plot variable importance from different algorithsm

def features_print2(predicted_fit, training_var):
   indices = np.argsort(predicted_fit.feature_importances_)
   # plot as bar chart
   plt.barh(np.arange(len(training_var.columns)), predicted_fit.feature_importances_[indices])
   plt.yticks(np.arange(len(training_var.columns)) + 0.25, np.array(training_var.columns)[indices])
   _ = plt.xlabel('Relative importance')


#write a re-usable module to plot the inital random forests performance on 10-fold cross validation
#cross vallidation performance
def cross_validate_perf (df, cv,clf, target):
    #define inital true positive rate and the spacing for false positive rate
    mean_tpr = 0.0
    mean_tpr_train = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train, test) in enumerate(cv):
        #print (train, test)
        #create probability score for training and crossvalidation data set
        probas_training = clf.fit(df.drop(target,1).values[train], df.target.values[train]).predict_proba(df.drop(target,1).values[train])
        probas_ = clf.fit(df.drop(target,1).values[train], df.target.values[train]).predict_proba(df.drop(target,1).values[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(df.target.values[test], probas_[:, 1])
        fpr_train, tpr_train, thresholds_train = roc_curve(df.target.values[train], probas_training[:, 1])
        #get the average ptc by using interpolation for each iteration
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr_train += interp(mean_fpr, fpr_train, tpr_train)        
        mean_tpr[0] = 0.0
        mean_tpr_train[0] = 0.0
        roc_auc = auc(fpr, tpr)
        roc_auc_train = auc(fpr_train, tpr_train)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    #plot the random line 
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    
    #calculate the mean of the tpr for both training and test data
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_tpr_train /= len(cv)
    mean_tpr_train[-1] = 1.0
    mean_auc_train = auc(mean_fpr, mean_tpr_train)
    #plot the average curve
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (AUC = %0.2f)' % mean_auc, lw=2)
    plt.plot(mean_fpr, mean_tpr_train, 'k--',label='Mean Training ROC (AUC = %0.2f)' % mean_auc_train, lw=2, color='red')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()



# Module to tune the machine learning algorithm using grid search. 
#This is a brute force method to check every combination of the method.
def gridsearch (param_test, clf, df, target):
    gsearch1 = GridSearchCV(estimator = clf, 
    param_grid = param_test, scoring='roc_auc',n_jobs=4,iid=False, cv=10)
    gsearch1.fit(df.drop(target,1),df[target])
    return (gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)


#module to check the performance of the test dataset by using the model from training data.
def test_perf (df, test,clf, target):
    
    mean_tpr = 0.0
    mean_tpr_train = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []

    probas_training = clf.fit(df.drop(target,1).values, df.target.values).predict_proba(df.drop(target,1).values)
    probas_ = clf.fit(df.drop(target,1).values, df.target.values).predict_proba(test.drop(target,1).values)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(test.target.values, probas_[:, 1])
    fpr_train, tpr_train, thresholds_train = roc_curve(df.target.values, probas_training[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr_train += interp(mean_fpr, fpr_train, tpr_train)        
    mean_tpr[0] = 0.0
    mean_tpr_train[0] = 0.0
    roc_auc = auc(fpr, tpr)
    roc_auc_train = auc(fpr_train, tpr_train)
    plt.plot(fpr, tpr, lw=1, label='ROC Test(AUC = %0.2f)' % ( roc_auc))
    plt.plot(fpr_train, tpr_train, lw=1, label='ROC Train (AUC = %0.2f)' % ( roc_auc_train))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()




#module to compare multiple classifier in the array clf
def cross_validate_perf_vs (clf, df,cv, target):
    

    for clf in clf:
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []
        for i, (train, test) in enumerate(cv):
            print (train, test)
            probas_ = clf.fit(df.drop(target,1).values[train], df.converted.values[train]).predict_proba(df.drop(target,1).values[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(df.converted.values[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        
        
        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, label='%s ROC (AUC = %0.2f)' % (str(clf)[0:str(clf).find("(")],mean_auc), lw=1)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison')
        plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.show()
    



#this is the module to tune the paremeters of any machine learning module

def validation_param (clf, param_name, param_range, x, y):
    param_range = [1,5,20]
    param_name="n_estimators"
    train_scores, test_scores = validation_curve(
        RandomForestClassifier ( ), df1.drop(target,1), df1[target], param_name="n_estimators", param_range=param_range)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title("Validation Curve")
    plt.xlabel("%s" %param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.xlim(min(param_range), max(param_range))
    plt.plot(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()





#this is the module to print cart conditions and change the color of the end node

def print_cart_tree (d0, d1, level, min_leaf, cutoff, color):
    
    # train the CART model
    cart=tree.DecisionTreeClassifier(random_state=0,min_samples_leaf=int('%d' % (min_leaf)), max_depth=int('%d' % (level)))
    cart=cart.fit(d0, d1)
    #set up all relavant arrays with the branches, threshold, features and values
    left      = cart.tree_.children_left
    right     = cart.tree_.children_right
    threshold = cart.tree_.threshold
    features  = [d0.columns[i] for i in cart.tree_.feature]
    value = cart.tree_.value
    
    # pick all indexes for all the terminal nodes, that is left or right with -1
    indexes = range(len(left))
    
    d=0
    # start of the while loop
    while d<len(left):
        # set up the threshold 
        if float(value[d][0][1]/(value[d][0][1]+value[d][0][0])) >= float('%f' % (cutoff)):
            ins=0
            #starting with each terminal node
            aaa=indexes[d]
            #print ("level %s , leaf node" % (level) + str(indexes[d]) + " and set:" + str(d))
            while aaa>0:
                #if the index is not in the left branch
                if [i for i,x in enumerate(left) if x == aaa]==[]:     
                    #print the  condition with the > sign
                    print  (" " + str(features[right.tolist().index(aaa)]) + " > " + str(threshold[right.tolist().index(aaa)]))
                    aaa=right.tolist().index(aaa)
                    ins=ins+1
                    #if it is the terminal node, print the values and fraud rate
                    if aaa==0:
                        print (str(value[indexes[d]]))
                        a=value[indexes[d]][0][1]/(value[indexes[d]][0][1]+value[indexes[d]][0][0])*100
                        print (("%.2f" % a) + "%")
                        print ("This is a level " + str(ins) + " branch")
                        print ("")
                    else:
                        print (" and ")
                else:
                    #if the condition is in the left branch, print the condition with <= sign
                    print  (" " + str(features[left.tolist().index(aaa)]) + " <= " + str(threshold[left.tolist().index(aaa)]) )
                    aaa=left.tolist().index(aaa)
                    ins=ins+1
                    if aaa==0:
                        print (str(value[indexes[d]]))
                        a=value[indexes[d]][0][1]/(value[indexes[d]][0][1]+value[indexes[d]][0][0])*100
                        print (("%.2f" % a) + "%")
                        print ("This is a level " + str(ins) + " branch")
                        print ("")
                    else:
                        print (" and ")
        d=d+1    
        
    import re
    from IPython.display import Image  
    from sklearn.externals.six import StringIO
    import pydotplus
    import pydot
    dot_data = StringIO()  
    tree.export_graphviz(cart, out_file=dot_data,feature_names=d0.columns ,filled=True,rounded=True  )  
   
    v= dot_data.getvalue() 
    tt=[]
    aaaa= dot_data.getvalue()
    pos=0
    result = re.search('nvalue = \[(.*)\]", fill', aaaa[pos:])
    while result is not None:
        result = re.search('nvalue = \[(.*)\]", fill', aaaa[pos:])
        if result is not None:
            tt.append([int(i) for i in result.group(1).split(",")])
        aaaa=aaaa[pos:]
        pos =(aaaa.find('nvalue'))+6
        
    dd=[m.end() for m in re.finditer('fillcolor="', v)]
    
    gini=[[m.start(),m.end()] for m in re.finditer('nvalue = \[(.*)\]"', v)]    
    i=len(tt)-1    
    while i>=0:
        if float(tt[i][1])/(tt[i][0]+tt[i][1])>=cutoff:
            v=v[:dd[i]]+color+v[dd[i]+9:]
        v=v[:gini[i][1]-1]+"\\nevent_rate = " + str(round(float(tt[i][1])*100/(tt[i][0]+tt[i][1]),2))+"%" + v[gini[i][1]-1:]
        i=i-1
        

    graph = pydotplus.graph_from_dot_data(v)  
    graph.write_png('tree.png')
    

