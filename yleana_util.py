import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def groupByStudentTypeTopic(df):
    studentsDF = df[['firstName','lastName','type','topic','correct']]
    grouped = studentsDF.groupby(['firstName','lastName','type','topic'],sort=True)
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
    r = groupedDF.pivot('firstName','type',agg)
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

def getTopicWeight(df):
    '''
    Calculate the relative weight of each topic within its subject area
    args:
        df: original cleaned data frame
    returns:
        topicsDF: a dataframe with type, topic, and weight of that topic
    '''
    #only considering full tests, so no 'BB's
    d0 = drop_test(df,'BB')
    #YL_6_PP_SAT_S0111 doesn't have topics
    d0=drop_test(d0,'YL_6_PP_SAT_S0111')

    #first get num of questions per topic for each test for each student
    d1 = groupData(d0,['firstName','testID','type','topic'],'correct')
    d1.rename(columns={'size':'questionsPerTopic'}, inplace=True)

    #then get the average number of questions per topic for each test
    d2 = groupData(d1,['type','topic'],'questionsPerTopic')
    d2.rename(columns={'mean':'meanQsPerTopic'}, inplace=True)

    #Get number of questions per test
    tt1 = d0.groupby(['testID','type','testQuestionNumber','testSectionNumber']).agg(np.size)
    tt1=tt1.reset_index()
    tt1 = tt1[['testID','type','testQuestionNumber','testSectionNumber']]
    tt2 = tt1.groupby(['testID','type']).agg([np.size])['testQuestionNumber']
    tt2.rename(columns={'size':'questionsPerTestType'}, inplace=True)
    testQs=tt2.reset_index()

    #Get number of topic appearances per type to establish topic weight (questions)
    d3 = groupData(d2,['type'],'meanQsPerTopic')
    d3 = d3.rename(columns={'sum':'topics_x_Qs'})
    d3 = d3[['type','topics_x_Qs']]

    d2 = pd.merge(d2,d3,how='left',on=['type'])
    d2['topicWeight'] = d2['meanQsPerTopic']/d2['topics_x_Qs']
    d2.sort(['type','topicWeight'],ascending=False)
    topicsDF = d2[['type','topic','topicWeight']]
    return topicsDF

def drawTopicsBarChart(cleanDF,subject,title,n_topics=10):
    '''
    Draw a bar chart of topic weights, to get a sense of the broadest topics
    '''
    topicsDF = getTopicWeight(cleanDF)
    df = topicsDF[topicsDF['type']==subject].sort('topicWeight',ascending=False).head(n_topics)
    df = df.sort('topicWeight')
    
    #size and position of bars
    bar_pos = np.arange(df.shape[0])
    bar_size = df['topicWeight']
    bar_labels = df['topic'].tolist()

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
    Get score by topic, by student, along with whether the score is above a passing threshold
    args:
        df: clean dataframe
        columns: columns to group by
        statVar: statistic your'e measuring
        passingThreshold: minimum score to pass
    returns: 
        studentPerf: dataframe listing scores by student by topic
    '''
    foo = groupData(df,columns,statVar)
    foo['wrong']=foo['size'] - foo['sum']

    #establish passing score
    foo['passing']=0
    foo.loc[foo['mean']>=passingThreshold,'passing']=1
    foo.rename(columns={'size':'numQuestions','sum':'numCorrect','mean':'score'},inplace=True)
    studentPerf = foo
    
    return studentPerf
    
def compareToClass(df,columns=['firstName','type','topic'],statVar='correct',passingThreshold=0.5):
    studentPerf = getPerfByStudent(df,columns,statVar,passingThreshold)
    foo = studentPerf[['type','topic','passing']]
    grouped = foo.groupby(by=['type','topic'],as_index=True)
    bar = grouped.agg([np.size,np.sum, np.mean])['passing']
    bar = bar.reset_index()
    bar.rename(columns={'size':'numStudentsGivenTopic','sum':'numStudentsPassed','mean':'pctStudentsPassed'},inplace=True)
    return studentPerf,bar

def getClassAvg(df,columns=['firstName','type','topic'],statVar='correct',passingThreshold=0.5):
    '''
    get the class average
    '''
    studentPerf = getPerfByStudent(df,columns,statVar,passingThreshold)
    foo = studentPerf[['type','topic','score']]
    grouped = foo.groupby(by=['type','topic'],as_index=True)
    bar = grouped.agg([np.size, np.mean])['score']
    bar = bar.reset_index()
    bar.rename(columns={'size':'numStudentsGivenTopic','mean':'classAvg'},inplace=True)
    return studentPerf,bar

def buildFocusTable(df,studentID,testID,subject,passingThreshold=0.6,minWrong=5):
    '''
    Get a data frame of topics in which this student is farthest behind the rest of the class, weighted by topic weight.
    These are recommendations for further study
    args:
        df: raw dataframe
        testID: test from which you want to build a recommendation table
        studentID:student ID integer
        subject: math, reading, sentence, or writing
        passingThreshold: minimum score to pass 
        minWrong: minimum number of wrong answers to make a recommendation
    returns:
        rec: Dataframe of topics in which this student is farthest behind the rest of the class,
                ranked by the difference between this student's % correct and the class avg.
    '''
    
    #optionally specify a testID, otherwise use all tests
    if testID is not None:
        df = df.loc[df['testID']==testID,:].copy()
    df = df.loc[df['type']==subject,:]
        
    #list of topics
    topicsDF = getTopicWeight(df)
    
    #get student and class performance on each topic
    studentPerf,classPerf = getClassAvg(df,columns=['studentID','type','topic'],statVar='correct',passingThreshold=passingThreshold)
    classPerf = classPerf.sort('numStudentsGivenTopic')
    q1 = pd.merge(studentPerf,classPerf,on=['type','topic'])
    q2 = pd.merge(q1,topicsDF, how='left',on=['type','topic'])
    rec = q2[['studentID','type','topic','topicWeight','wrong','score','classAvg']].copy()
    rec['scoreDiff']=rec['score'] - rec['classAvg']
    rec['weightedScoreDiff'] = rec['scoreDiff']*rec['topicWeight']
    
    #only recommend areas where the student got at least a few wrong
    rec = rec[rec['wrong']>=minWrong]
    
    for col in ['topicWeight','score','classAvg','scoreDiff']:
        rec[col] = rec[col].round(2)
    rec.sort('weightedScoreDiff',ascending=True, inplace=True)
    return rec.loc[rec['studentID']==studentID].head()

def getTrendsOverTime(df,testString,columns=['type','testNum']):
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
    subjects = trendsDF.type.unique()
    for subject in subjects:
        foo = trendsDF[trendsDF['type']==subject]
        x = foo['testNum']
        y = foo['avgScore']
        plt.plot(x,y,'-o')
        plt.label=subject

    plt.ylabel('Avg Score')
    plt.xlabel('Test Number')
    plt.title(title)
    plt.legend(subjects,bbox_to_anchor=(1, 1), loc=2)

def groupTopics(df,statsDF,columns,statVar):
    '''
    Group the data by topic
    '''
    subDF = df[columns+[statVar]]
    grouped = subDF.groupby(columns,sort=True)
    groupedDF = grouped.agg([np.size,np.sum, np.mean,np.std])[statVar]
    groupedDF = groupedDF.reset_index()
    groupedDF.rename(columns={'mean':'meanNumTopics'}, inplace=True)
    mergedDF = pd.merge(statsDF, groupedDF[columns+['meanNumTopics']], how='left', on=columns)
    return mergedDF

def getMostWrongs(df,subject):
    '''
    given the dataframe and subject, identify the topics with the most wrong answers
    '''
    if subject:
        subjDF = df.loc[df['type']==subject]
    else: subjDF = df.copy()
    statsDF = groupData(subjDF,['type','topic','firstName'],'correct')
    mg = groupTopics(df,statsDF,['topic'],'numTopics')
    mg['wrong']=mg['size'] - mg['sum']
    return mg.sort('wrong',ascending=False)