# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:34:31 2015

@author: NDT567
"""

# -*- coding: utf-8 -*-
""" 
Created on Mon Jun 08 13:32:34 2015

@author: NDT567


This is the second main program.
The purpose is to evaluate the best algorism (gradient boosting from program 1) on a dynamic peroid.

To evaluate the performance of the dynamic modelling, I am using the continuous confusion matrix,
which is the sum of all model outputs from a dynamic period

"""



import re
import pyodbc
conn = pyodbc.connect('DRIVER={Teradata};DBCNAME=oneview;UID=ndt567;PWD=;')  
import pydot
from sklearn import metrics 
from sklearn.ensemble import GradientBoostingClassifier
import middle_out as mi
from middle_out import *

#initialize list, set up target variable, a time period to move and the time inverval for each model
zz={}
num=60
num2=360
interval=30
interval2=30
target='isfrd'
d_append= pd.DataFrame()
d_append_all=pd.DataFrame()
#initialize model paremeters
estimator=200
learn_rate=0.1

#the start of the for loop 
while num>0:    
#training dataset    
    sql = ''' 
    sel 
    acct_id,
    case when PRI_CBR_MATCH_LEVEL_HOME_PHONE ='' then null else
    PRI_CBR_MATCH_LEVEL_HOME_PHONE end as PRI_CBR_MATCH_LEVEL_HOME_PHONE ,
    PRI_REPORTED_INCOME_AMT       	,
    PRI_MONTHS_RESIDENCE_CNT      	,
    cast(APP_TYPE as varchar(1)) APP_TYPE                    	,
    credit_limit_assigned,
    case when PRI_CBR_MATCH_LEVEL_SSN ='' then null else
    PRI_CBR_MATCH_LEVEL_SSN end as PRI_CBR_MATCH_LEVEL_SSN ,
    channel                       	,
    CURR_ADDRESS_STATE            	,
    CASH_ADVANCE_IND              	,
    BUREAU_HIT_TU                 	,
    BUREAU_HIT_EQI                	,
    oldest_trade_age_mos          	,
    tot_num_inq_excl              	,
    bureau_file_age_yrs           	,
    TOT_NUM_BKRPT_ON_RECORD       	,
    sin_valid                     	,
    case when isfrd=0 then 0
    when isfrd=1 and  date-%d>=lsrp_dt then 1
    when isfrd=1 and  date-%d<lsrp_dt then 0
    else 0
    end as isfrd ,
    case when OPEN_DT>=date-%d - %d*2 then 50
    when OPEN_DT>=date-%d - %d*6 and OPEN_DT<date-%d - %d*2 then 30
    else 10 end as relav
                           	
    from ud155.u31308_frd_risk_var_tbl
    where credit_limit_assigned>=1100         
    and OPEN_DT>=date-%d - %d and OPEN_DT<date-%d
    ''' % (num, num, num, interval2, num, interval2, num, interval2,num, num2,num)
    
#validation dataset    
    sql1 = ''' 
    sel 
    acct_id,
    case when PRI_CBR_MATCH_LEVEL_HOME_PHONE ='' then null else
    PRI_CBR_MATCH_LEVEL_HOME_PHONE end as PRI_CBR_MATCH_LEVEL_HOME_PHONE ,
    PRI_REPORTED_INCOME_AMT       	,
    PRI_MONTHS_RESIDENCE_CNT      	,
    cast(APP_TYPE as varchar(1)) APP_TYPE,
    credit_limit_assigned,
    case when PRI_CBR_MATCH_LEVEL_SSN ='' then null else
    PRI_CBR_MATCH_LEVEL_SSN end as PRI_CBR_MATCH_LEVEL_SSN ,
    channel                       	,
    CURR_ADDRESS_STATE            	,
    CASH_ADVANCE_IND              	,
    BUREAU_HIT_TU                 	,
    BUREAU_HIT_EQI                	,
    oldest_trade_age_mos          	,
    tot_num_inq_excl              	,
    bureau_file_age_yrs           	,
    TOT_NUM_BKRPT_ON_RECORD       	,
    sin_valid                     	,
    isfrd       ,
    50 as relav
    from ud155.u31308_frd_risk_var_tbl
    where credit_limit_assigned>=1100         
    and OPEN_DT>=date-%d and OPEN_DT<date-%d+%d

    ''' % ( num, num, interval)
    
    #load the training and validation dataset into pandas dataframe
    df = pd.DataFrame()
    df= pd.concat([df, pd.read_sql(sql,conn)], axis=0)
    
    df_val=pd.DataFrame()
    df_val= pd.concat([df_val, pd.read_sql(sql1,conn)], axis=0)

# run the middle out module to convert the data into ML datasets
    
    d00,d00_val =mi.middle_out(df,df_val)
    d00=mi.down_size (d00, 'isfrd', 85, 15)
#drop the nulls and separate the datasets with denpendent v.s. indenpendent variables
    d00 = d00.dropna()
    d00_val=d00_val.dropna()

    d1_val=d00_val['%s'  % target].values

# fit the GB model    
    clf=GradientBoostingClassifier (n_estimators= int('%d' % estimator), learning_rate= float("%.2f" %learn_rate),  random_state=0 )
    clf=clf.fit(d00.drop(['%s' % target, 'acct_id'],1), d00['%s' % target].values)
    ########################
    
# fit the model and store the result into a dictionary for each iteration    
    expected_val = d1_val
    predicted_val=clf.predict(d00_val.drop(['%s' % target, 'acct_id'],1))
    predicted_val_prob=clf.predict_proba(d00_val.drop(['%s' % target, 'acct_id'],1))[:,1]
    metrics.confusion_matrix(expected_val, predicted_val)
    zz[num]=(metrics.confusion_matrix(expected_val, predicted_val))
    d00_val['pred']=predicted_val
    d00_val['pred_prob']=predicted_val_prob
    d_append= pd.concat([d_append, d00_val.loc[d00_val['pred']==1][['acct_id','isfrd','pred']]], axis=0)
    d_append_all=pd.concat([d_append_all, d00_val[['acct_id','isfrd','pred','pred_prob']]], axis=0)
    num=num- int('%d' % (interval))

 ##################

# combine all iterations from the dynamic model and append it to one CCM dataset
v=[]
for k in zz:
    v.append(numpy.array((numpy.array(zz[k].flatten()).tolist()) +[k]))

    
CCM = pd.DataFrame(v, columns=('true_pos', 'false_neg', 'false_pos', 'true_neg','days'))
CCM=CCM.sort(columns='days')
CCM.sum()


quantile_plot(d_append_all, 'pred_prob', 10,'isfrd')

plo=d_append_all.loc[d_append_all['pred'] ==1]

quantile_plot(plo, 'pred_prob', 10,'isfrd')


export_CCM (zz, 'CCM.xlsx')

writer = pd.ExcelWriter('output.xlsx')
d_append_all.to_excel(writer,'Sheet1')

writer.save()








