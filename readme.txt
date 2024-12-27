-------------------------------------
Magazine Recommendation system Readme
-------------------------------------

Section 1 below is about our deployment dashboard link, administrator and user accounts. If you want to setup your environment, you can go to Section 2.

Section 1:
------------------------------------------------------------------------------------------------
Link - https://bi66yz.onrender.com/
Admin account: 
Login ID: admin
Password: 1234

Demo Account 1:
Login ID: AFERCDY2EFJKT7QUQ75GISNHTFOQ
Password: 1234

Demo Account 2:
Login ID: AGCK7OO3T5VA4XRDC2IR4LETU57A
Password: 1234

Demo Account 3:
Login ID: AE3HMAARJIZHRMXABXVRSU3KJSPA
Password: 1234
------------------------------------------------------------------------------------------------

Section 2:
------------------------------------------------------------------------------------------------
The following steps are used to install and execute our recommendation system. Please try to follow. Thank you!

Step 1: Download the "Assignment2_RS_239669700_bi66yz.zip" file from the University of Sunderland / Github and unzip it to the "C:\" drive.

Step 2: Confirm that there are "Assignment2_for_magazine_rs_program_239669700_bi66yz.py" and "dashboard_rs.py" program files in the "C:\Assignment2_RS_239669700_bi66yz" folder

Step 3: Create "datasets", "log" and "export_model" folders in the "Assignment2_RS_239669700_bi66yz" folder.

Step 4: Download both datasets from the links below.
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Magazine_Subscriptions.jsonl.gz
https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Magazine_Subscriptions.jsonl.gz

Step 5: Unzip the two dataset files, and then copy the "Magazine_Subscriptions.jsonl" and "meta_Magazine_Subscriptions.jsonl" files to the "C:\Assignment2_RS_239669700_bi66yz\datasets\" folder.

***************************
Remark: If you have not installed python or do not have much experience, you must follow Step 6 to Step 10, otherwise jump to Step 11
***************************

Step 6: Download and install Anaconda from the link below
https://docs.anaconda.com/anaconda/install/

Step 7: Click the Start menu, type “cmd” or “Command Prompt,” and press Enter.

Step 8: Go to our program folder, execute the following command in the command prompt.
cd C:\Assignment2_RS_239669700_bi66yz\

Step 9: To create a python virtual environment, execute the following command in the command prompt.
python -m venv C:\Assignment2_RS_239669700_bi66yz\

Step 10: To start this virtual environment, execute the following command in the command prompt.
C:\Assignment2_RS_239669700_bi66yz\Scripts\activate

Step 11: Install the following python library and execute the following command lines line by line in the command prompt.
pip install notebook
pip install numpy==1.24.4
pip install pandas==2.0.3
pip install matplotlib
pip install scikit_surprise==1.1.4
pip install dash
pip install dash_bootstrap_components
pip install plotly
pip install nltk
pip install wordcloud

Step 12: Check whether python has the above library installed and execute the following command in the command prompt.
pip list

***************************
Remark: You can download our model directly from https://github.com/kiukwok/bi66yz_rs/raw/refs/heads/master/export_model/surprise_model and
copy it to the "export_model" folder, otherwise you must run Step 13, this program takes about 30 mins to run
***************************

Step 13: Training model program, execute the following command in the command prompt.
python C:\Assignment2_RS_239669700_bi66yz\Assignment2_for_magazine_rs_program_239669700_bi66yz.py

Step 14: Check whether the program exported the "surprise_model" file to the "export_model" folder.

Step 15: Open the dashboard command and execute the following command in the command prompt.
python ./dashboard_rs.py

Step 16: Copy the URL "http://127.0.0.1:8888" and open this URL in the web browser

Step 17: Log in to the dashboard using the administrator account below.
Login ID: admin
Password: 1234

Step 18: The following demo user logins are used in the “End user Recommendation” section.
Demo Account 1:
Login ID: AFERCDY2EFJKT7QUQ75GISNHTFOQ
Password: 1234

Demo Account 2:
Login ID: AGCK7OO3T5VA4XRDC2IR4LETU57A
Password: 1234

Demo Account 3:
Login ID: AE3HMAARJIZHRMXABXVRSU3KJSPA
Password: 1234
------------------------------------------------------------------------------------------------