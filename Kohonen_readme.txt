We set learning rate is 0.5, since learning rate as 0.5 is fairly closed to somewhere in the middle, which means the rate won't be too large or small so we will avoid the data training drop to local minima, with proper epoch each time

termination criteria: it runs 1000 times and then stop after that

Inputs number: 3 this is because we have 3 dimensional data or 3 features set which is x, y, z respectively and each represents one node

Outputs number: 2 so we separate this to 2 cluster groups and is quit visible to see or achieve 

Data preprocessing: I get rid of first line which is specification of 3 sets of data when i read csv file and add data to a list. Then i convert them to float so we could calculate it in propagation way right. Finally, we normalized the data and shuffle the data, to get rid of gaps or big differences, and it won't be hard to train or train in the wrong way.
==========================================================================================
Initial weights of class1 and class2: [{'weights': [0.3012275720708666, 0.2247628558417898, 0.5541774542990323]}, {'weights': [0.5747944158108705, 0.6065675435868519, 0.3409436387952265]}]

Give each feature a random initial weight so that the function can be train without interference.

Error from Class1: 171.7026464377354
Error from Class2: 171.51915964809785
Total Error: 343.22180608583324

2 Final weight: [{'weights': [0.4924175231941025, 0.4792108066956434, 1.2198105481224784]}, {'weights': [0.4979531780964822, 0.49089009880076045, -0.21793116936373935]}]
