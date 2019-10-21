# Problem:  Indoor positioning with wireless signal

You may or may not be aware that accurate indoor positioning has high commericial value for the network operators. Imagine when you are desperately looking for a shopping store in a large shopping mall, you can hardly describe your precise position to your friends. In fact, accurate position can be widely used to assist car parking, robot collaboration, failure detection of pipes and wires.  However, the precise positioning is quite difficult in the indoor environment when you only have wireless signals. In the indoor environment, the wireless signals are much lessly predictable because of the multi-path effect. So, we don't have very good approach to improve the positioning accuracy until AI and ML steps in. It can offer a computationally afforable solution to balance the number of equipment and the desired accuracy.  In this contest, you are expected to use machine learning to predict the postion inside a building. The data comes from the real customer site and they are tailored to fit for the purpose of excerise. We wish you can enjoy the experience of solving the real challenge and we expect to see your wisdom and good achievement.

# Proposed Solution methodology:

By learning the RSRP(Reference Signal Receiving Power) signal strength signature, you are expected to predict the position from the new RSRP signatures. Your machine learning application will learn the RSRP data from different cells at training time and will generate the predicted postion in (x,y) format from the test RSRP data. The model quality will be measured by the mean square error and error distribution of your prediction. On top of model quality ranking, the selected experts from Beijing TC will review your team's implementation and your team will also be invited to introduce your solution. The final ranking will be decided by the sum of scores of model quality, code quality and presentation quality. Good luck to all the teams.

# Dataset:

You will get training data set "train.csv", in which x and y reprecent position(x, y), value in other columns is RSRP for each cell.

We sample a room every 20cm, and make a grid of 156*317. On every grid point, we record the signal strength signature from 6 BTS in 2 bands (2.1G and 3.5G). So the columns contain the (x,y) as position and 12 numbers as the signal strength RSRP. The test file describes a route in the room, on every route point, there are 12 RSRP values. Your mission is to expect the route in the room.

"test.csv" is for you to exercise, after programming, we'll provide real test data.

 To find data set:

https://nokia.sharepoint.com/sites/CodeRallyBeijing/Shared%20Documents/CodeRally2019/Competition_Datasets

You can also find the data set at Kaggle:
https://www.kaggle.com/hzhao011/code-rally-2019

# How to build
0. yum install gcc
   yum install zlib-devel
   yum install libffi-devel
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python get-pip.py
   yum install openssl
   yum install openssl-devel

1. python3.7 download and install
   https://www.python.org/ftp/python/3.7.3/Python-3.7.3.tgz

   tar xvf Python-3.7.3.tgz
   ./configure
   make
   make altinstall

   /usr/local/bin/python3.7
   /usr/local/lib

2. pip install scikit-learn

3. python3.7 indoorpospred.py . remove not
 
