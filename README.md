# yleana
Analyzing Yleana Practice Tests

#Instructions

##Git Setup
1. Install Homebrew

	```ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"```
1. Install Git

	```brew update```
	
	```brew install git```
	
1. Go to a local directory where you want this kept

	``` cd [wherever you want]```

1. Git clone

	``` git https://github.com/charlesdguthrie/yleana.git ```

##Directory Setup
create four directories:

1. data
2. reports
3. plots
4. scores_by_concept

Within reports and scores_by_concept, create a directory for each test ID.  


##Adding a New Test
1. get csv file from semi
1. put the file in data
1. create a directory for the test in reports and scores_by_concept
1. Go into data prep, and add an entry to the dictionary (test name: date)
1. Go into the main function in score_report.py and change the filename, the last test ID, and the test ID
1. Run it
1. Upload to web via filezilla


 