'''
data_prep.py: All the functions for preparing and cleaning data

author:Charlie Guthrie
'''
import pandas as pd
import numpy as np
import yleana_util as yp
import datetime

#TODO: make this import a csv file
def addDates(df):
    '''
    merge test dates into df
    '''
    test_dates = {"testID":["YL_2_PP_SAT_S0112","YL_1_PP_SAT_S0114"],
                  "testDate":[datetime.datetime(2015,7,5),datetime.datetime(2015,6,26)]
                 }
    datesDF = pd.DataFrame.from_dict(test_dates)
    return pd.merge(df,datesDF)

def countNullQuestions(df):
    '''
    removes test questions with a null correct answer
    prints a warning to share the number of null questions
    '''
    columns = ['testID','testQuestionNumber','testSectionNumber']
    nullsDF = df.loc[(df['correctAnswer'].isnull()),columns].drop_duplicates()
    num_nulls = nullsDF.shape[0]
    
    if num_nulls>0:
        print "Warning: %i questions had a null value for the correct answer" % num_nulls

    return df.loc[df['correctAnswer'].notnull(),:]

def clean_data(df):
    '''
    changes column names, creates a column for correct answers
    args: raw dataframe
    returns: cleaned dataframe
    '''

    d = df.copy()
    d.rename(columns={'studentUniqueID':'studentID','testName':'testID','type':'subject','answer':'studentAnswer','CorrectAnswer':'correctAnswer','difficultyLevel':'difficulty'},inplace=True)
    
    #remove null correct answers.  Can't assess students there
    d = countNullQuestions(d)

    #create column for correct answers
    d['correct'] = 0
    d.loc[d['studentAnswer']==d['correctAnswer'],'correct']=1

    return d

def mapConcepts(df,conceptMapPath='data/concept_map.csv',badConcepts=['ZI','LP - long passage']):
    '''
    map the granular specific concepts to the broader set
    returns data frame with broader mapping
    '''
    cm = pd.read_csv(conceptMapPath)
    df2 = pd.merge(df,cm,how='left',on=['concept','subject'])
    
    #fill in blank broad concepts
    df2['broad_concept'].fillna(df2['concept'], inplace=True)
    
    #remove bad concepts
    df2.drop(df2.index[df2['broad_concept'].isin(badConcepts)],inplace=True)
    
    df2.drop('concept',axis=1,inplace=True)
    df2.rename(columns = {'broad_concept':'concept'},inplace=True)
    return df2

def addNumConcepts(rawDF):
    '''
    get the number of concepts associated with each question
    args:df
    returns: df with numConcepts attached
    '''

    df = rawDF.copy()
    qg = yp.groupData(df,['firstName','testID','testQuestionNumber','testSectionNumber'],'correct')
    qg2 = qg.groupby(['testID','testQuestionNumber','testSectionNumber']).max().reset_index()
    qg2.sort(['testID','testSectionNumber','testQuestionNumber','firstName'])
    qg2.rename(columns={'size':'numConcepts'}, inplace=True)
    df = pd.merge(df, qg2[['testID','testQuestionNumber','testSectionNumber','numConcepts']], how='left', on=['testID','testQuestionNumber','testSectionNumber'])
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

def main(fn, makeIDs, assignClass):
    rawDF = pd.read_csv(fn)
    df = clean_data(rawDF)
    df = addNumConcepts(df)
    if makeIDs:
        df = makeStudentIDs(df)
    if assignClass:
        df = assignToClass(df)
    return df

if __name__ == '__main__':
    FN = 'data/RawStudentDifficultyData.csv'
    main(FN, True,True)