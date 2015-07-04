import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def groupByStudentTypeConcept(df):
    studentsDF = df[['firstName','lastName','subject','concept','correct']]
    grouped = studentsDF.groupby(['firstName','lastName','subject','concept'],sort=True)
    return grouped.agg([np.size,np.sum, np.mean,np.std])

def groupData(df,columns,statVar):
    '''
    group the data by the columns, aggregating the statVar with sum, mean, std
    args:
        df: dataframe
        columns:columns to group by
        statVar: column to aggregate.  
    '''
    subDF = df[columns+[statVar]]
    grouped = subDF.groupby(columns,sort=True)
    groupedDF = grouped.agg([np.size,np.sum, np.mean])[statVar]
    groupedDF = groupedDF.reset_index()
    return groupedDF

def drawHeatmap(df,columns,agg,sortingColumn):
    groupedDF = groupData(df,columns,'correct')
    r = groupedDF.pivot('firstName','subject',agg)
    r = r.sort(sortingColumn)

    #sns.set_context(context)
    f=sns.heatmap(r, annot=True)
    f.set_title("Performance by Subject")
    f.set_xlabel("Student")
    f.set_ylabel("Subject")
    return r

def drop_test(df,desc):
    df2 = df.drop(list(np.where(df.testID.str.contains(desc))[0]))
    return df2.reset_index()

def getConceptWeight(df):
    '''
    Calculate the relative weight of each concept within its subject area
    args:
        df: original cleaned data frame
    returns:
        conceptsDF: a dataframe with subject, concept, and weight of that concept
    '''
    #only considering full tests, so no 'BB's
    d0 = drop_test(df,'BB')
    #YL_6_PP_SAT_S0111 doesn't have concepts
    d0=drop_test(d0,'YL_6_PP_SAT_S0111')

    #first get num of questions per concept for each test for each student
    d1 = groupData(d0,['firstName','testID','subject','concept'],'correct')
    d1.rename(columns={'size':'questionsPerConcept'}, inplace=True)

    #then get the average number of questions per concept for each test
    d2 = groupData(d1,['subject','concept'],'questionsPerConcept')
    d2.rename(columns={'mean':'meanQsPerConcept'}, inplace=True)

    #Get number of questions per test
    tt1 = d0.groupby(['testID','subject','testQuestionNumber','testSectionNumber']).agg(np.size)
    tt1=tt1.reset_index()
    tt1 = tt1[['testID','subject','testQuestionNumber','testSectionNumber']]
    tt2 = tt1.groupby(['testID','subject']).agg([np.size])['testQuestionNumber']
    tt2.rename(columns={'size':'questionsPerTestType'}, inplace=True)
    testQs=tt2.reset_index()

    #Get number of concept appearances per subject to establish concept weight (questions)
    d3 = groupData(d2,['subject'],'meanQsPerConcept')
    d3 = d3.rename(columns={'sum':'concepts_x_Qs'})
    d3 = d3[['subject','concepts_x_Qs']]

    d2 = pd.merge(d2,d3,how='left',on=['subject'])
    d2['conceptWeight'] = d2['meanQsPerConcept']/d2['concepts_x_Qs']
    d2.sort(['subject','conceptWeight'],ascending=False)
    conceptsDF = d2[['subject','concept','conceptWeight']]
    return conceptsDF

def drawConceptsBarChart(cleanDF,subject,title,n_concepts=10):
    '''
    Draw a bar chart of concept weights, to get a sense of the broadest concepts
    '''
    conceptsDF = getConceptWeight(cleanDF)
    df = conceptsDF[conceptsDF['subject']==subject].sort('conceptWeight',ascending=False).head(n_concepts)
    df = df.sort('conceptWeight')
    
    #size and position of bars
    bar_pos = np.arange(df.shape[0])
    bar_size = df['conceptWeight']
    bar_labels = df['concept'].tolist()

    #plot
    fig = plt.figure(figsize=[5,5])
    plt.barh(bar_pos,bar_size, align='center', alpha=0.4)
    plt.yticks(bar_pos, bar_labels)
    plt.xticks([],[])
    for x,y in zip(bar_size,bar_pos):
        plt.text(x+0.02*max(bar_size), y, '%.3f' % x, ha='left', va='center')
    plt.title(title)

def getPerfByStudent(df,columns,statVar,passingThreshold):
    '''
    Get score by concept, by student, along with whether the score is above a passing threshold
    args:
        df: clean dataframe
        columns: columns to group by - usually topic
        statVar: statistic you're measuring
        passingThreshold: minimum score to pass
    returns: 
        studentPerf: dataframe listing scores by student by concept
    '''
    foo = groupData(df,columns,statVar)
    foo['wrong']=foo['size'] - foo['sum']

    #establish passing score
    foo['passing']=0
    foo.loc[foo['mean']>=passingThreshold,'passing']=1
    foo.rename(columns={'size':'numQuestions','sum':'numCorrect','mean':'score'},inplace=True)
    studentPerf = foo
    
    return studentPerf
    
def compareToClass(df,columns=['firstName','subject','concept'],statVar='correct',passingThreshold=0.5):
    studentPerf = getPerfByStudent(df,columns,statVar,passingThreshold)
    foo = studentPerf[['subject','concept','passing']]
    grouped = foo.groupby(by=['subject','concept'],as_index=True)
    bar = grouped.agg([np.size,np.sum, np.mean])['passing']
    bar = bar.reset_index()
    bar.rename(columns={'size':'numStudentsGivenConcept','sum':'numStudentsPassed','mean':'pctStudentsPassed'},inplace=True)
    return studentPerf,bar

def getClassAvg(df,columns=['firstName','subject','concept'],statVar='correct',passingThreshold=0.5):
    '''
    get the class average
    '''
    studentPerf = getPerfByStudent(df,columns,statVar,passingThreshold)
    foo = studentPerf[['subject','concept','score']]
    grouped = foo.groupby(by=['subject','concept'],as_index=True)
    bar = grouped.agg([np.size, np.mean])['score']
    bar = bar.reset_index()
    bar.rename(columns={'size':'numStudentsGivenConcept','mean':'classAvg'},inplace=True)
    return studentPerf,bar

def buildFocusTable(df,studentID,testID,subject,passingThreshold=0.6,minWrong=5):
    '''
    Get a data frame of concepts in which this student is farthest behind the rest of the class, weighted by concept weight.
    These are recommendations for further study
    args:
        df: raw dataframe
        testID: test from which you want to build a recommendation table
        studentID:student ID integer
        subject: math, reading, sentence, or writing
        passingThreshold: minimum score to pass 
        minWrong: minimum number of wrong answers to make a recommendation
    returns:
        rec: Dataframe of concepts in which this student is farthest behind the rest of the class,
                ranked by the difference between this student's % correct and the class avg.
    '''
    
    #optionally specify a testID, otherwise use all tests
    if testID is not None:
        df = df.loc[df['testID']==testID,:].copy()
    df = df.loc[df['subject']==subject,:]
        
    #list of concepts
    conceptsDF = getConceptWeight(df)
    
    #get student and class performance on each concept
    studentPerf,classPerf = getClassAvg(df,columns=['studentID','subject','concept'],statVar='correct',passingThreshold=passingThreshold)
    classPerf = classPerf.sort('numStudentsGivenConcept')
    q1 = pd.merge(studentPerf,classPerf,on=['subject','concept'])
    q2 = pd.merge(q1,conceptsDF, how='left',on=['subject','concept'])
    rec = q2[['studentID','subject','concept','conceptWeight','wrong','score','classAvg']].copy()
    rec['scoreDiff']=rec['score'] - rec['classAvg']
    rec['weightedScoreDiff'] = rec['scoreDiff']*rec['conceptWeight']
    
    #only recommend areas where the student got at least a few wrong
    rec = rec[rec['wrong']>=minWrong]
    
    for col in ['conceptWeight','score','classAvg','scoreDiff']:
        rec[col] = rec[col].round(2)
    rec.sort('weightedScoreDiff',ascending=True, inplace=True)
    return rec.loc[rec['studentID']==studentID].head()

def getTrendsOverTime(df,testString,columns=['subject','testNum']):
    if testString=='OL':
        sequencePos = -1
    elif testString=='PP_SAT':
        sequencePos = 3
    ol = df[df.testID.str.contains(testString)]
    ol['testNum']=ol.testID.str[sequencePos]
    trendsDF = groupData(ol,columns,'correct')
    trendsDF.rename(columns={'mean':'avgScore'},inplace=True)
    return trendsDF

def plotTrends(trendsDF,title):
    subjects = trendsDF.subject.unique()
    for subject in subjects:
        foo = trendsDF[trendsDF['subject']==subject]
        x = foo['testNum']
        y = foo['avgScore']
        plt.plot(x,y,'-o')
        plt.label=subject

    plt.ylabel('Avg Score')
    plt.xlabel('Test Number')
    plt.title(title)
    plt.legend(subjects,bbox_to_anchor=(1, 1), loc=2)

def groupConcepts(df,statsDF,columns,statVar):
    '''
    Group the data by concept
    '''
    subDF = df[columns+[statVar]]
    grouped = subDF.groupby(columns,sort=True)
    groupedDF = grouped.agg([np.size,np.sum, np.mean,np.std])[statVar]
    groupedDF = groupedDF.reset_index()
    groupedDF.rename(columns={'mean':'meanNumConcepts'}, inplace=True)
    mergedDF = pd.merge(statsDF, groupedDF[columns+['meanNumConcepts']], how='left', on=columns)
    return mergedDF

def getMostWrongs(df,subject):
    '''
    given the dataframe and subject, identify the concepts with the most wrong answers
    '''
    if subject:
        subjDF = df.loc[df['subject']==subject]
    else: subjDF = df.copy()
    statsDF = groupData(subjDF,['subject','concept','firstName'],'correct')
    mg = groupConcepts(df,statsDF,['concept'],'numConcepts')
    mg['wrong']=mg['size'] - mg['sum']
    return mg.sort('wrong',ascending=False)