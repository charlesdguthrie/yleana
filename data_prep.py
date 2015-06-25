'''
data_prep.py: All the functions for preparing and cleaning data

author:Charlie Guthrie
'''
import pandas as pd
import numpy as np
import yleana_util as yp

def clean_data(df):
    '''
    changes column names, creates a column for correct answers
    args: raw dataframe
    returns: cleaned dataframe
    '''
    d = df.copy()
    d.rename(columns={'name.1':'topic'},inplace=True)
    d.rename(columns={'name':'testID'},inplace=True)
    
    #create column for correct answers
    d['correct'] = 0
    d.loc[d['Studentsanswer']==d['CorrectAnswer'],'correct']=1
    
    return d

def addNumTopics(rawDF):
    '''
    get the number of topics associated with each question
    args:df
    returns: df with numTopics attached
    '''
    df = rawDF.copy()
    qg = yp.groupData(df,['firstName','testID','testQuestionNumber','testSectionNumber'],'correct')
    qg2 = qg.groupby(['testID','testQuestionNumber','testSectionNumber']).max().reset_index()
    qg2.sort(['testID','testSectionNumber','testQuestionNumber','firstName'])
    qg2.rename(columns={'size':'numTopics'}, inplace=True)
    df = pd.merge(df, qg2[['testID','testQuestionNumber','testSectionNumber','numTopics']], how='left', on=['testID','testQuestionNumber','testSectionNumber'])
    return df

def makeStudentIDs(df,index_column_list=['firstName','lastName']):
    '''
    Assign students unique ids based on the chosen index columns (default firstName, lastName)
    returns: df with student ids
    '''
    unique_students = df.drop_duplicates(index_column_list)
    unique_students.reset_index(inplace=True)
    unique_students = unique_students.loc[:,index_column_list]
    unique_students.loc[:,'studentID']=unique_students.index
    
    df2 = pd.merge(df,unique_students,on=index_column_list)
    
    return df2

def assignToClass(rawDF):
	df = rawDF.copy()
	df.loc[df['firstName'].isin(['Aeson','Ahna','Akayla','Allan','Alondra']),'class']="A"
	df.loc[df['firstName'].isin(['Amanda', 'Ashli',
       'Auston', 'Ayanna', 'Cheyanne', 'Clementina']),'class']="B"
	return df

#Define classes
def createClass(df,students,className):
    df.loc[df['firstName'].isin(students),'class']=className
    return df.loc[df['class']==className,:]

def main(fn):
	rawDF = pd.read_csv(fn)
	df = clean_data(rawDF)
	df = addNumTopics(df)
	df = makeStudentIDs(df)
	df = assignToClass(df)
	return df

if __name__ == '__main__':
    FN = 'data/RawStudentDifficultyData.csv'
    main(FN)