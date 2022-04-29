# Import Libraries
from functools import total_ordering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy

#reading the data and printing total shape of dataset

dataset = pd.read_csv("/Users/cvkrishnarao/Desktop/Intro_Programming_Assignment/5010_Project/Dataset/posturalDatset.csv")
X = dataset.iloc[:, 1:].values
Rows=dataset.shape[0]

# print(X)
attribute = ['tag', 'x', 'y', 'z']

#Let's Define a class
# that refers an objects to value decisions and childs as fields for attribute

class Node(object):
    def __init__(self):
        self.value = None
        self.decision = None
        self.childs = None



#function for finding entropy
"""#### This function defines a findEntropy functions which takes two arguments (data and rows) and
further defines the classified attribute (features) Activity columns and calculates the
entropy return the value of entrop and ans P(E). It finally selects the attribute based on feature(Activity)
which has the smallest entropy"""
def findEntropy(data, rows):
    walking=0
    falling=0
    l_down=0
    s_down=0
    sitting=0
    standing_up_from_lying=0
    on_all_fours=0
    sitting_on_the_ground=0
    standing_up_from_sitting=0
    standing_up_from_sitting_on_ground=0
    lying=0
    ans = -1
    idx = len(data[0]) - 1
    entropy = 0
    total_count=0
    # each iteration defined for the feature set choosing each value to match up with given Activity
    for i in rows:
        if data[i][idx] == 'walking':
            walking+= 1
        elif data[i][idx] == 'falling':
            falling+=1
        elif data[i][idx] == 'lying down':
            l_down+=1
        elif data[i][idx] == 'sitting down':
            s_down+=1
        elif data[i][idx] == 'sitting':
            sitting+=1
        elif data[i][idx] == 'standing up from lying':
            standing_up_from_lying+=1
        elif data[i][idx] == 'on all fours':
         on_all_fours+=1
        elif data[i][idx] == 'sitting on the ground':
         sitting_on_the_ground+=1
        elif data[i][idx] == 'standing up from sitting':
            standing_up_from_sitting+=1
        elif data[i][idx] == 'standing up from sitting on the ground':
            standing_up_from_sitting_on_ground+=1
        else:
            lying+=1
      # total_count based upon the activities
    total_count+=(walking+falling+l_down+s_down+sitting+standing_up_from_lying+sitting_on_the_ground+standing_up_from_sitting+standing_up_from_sitting_on_ground)

    if total_count==0:
        return 0,0
    #counting total_count division for each of the activities based upon the entropy
    a=walking/total_count
    b=falling/total_count
    c=l_down/total_count
    d=s_down/total_count
    e=sitting/total_count
    f=standing_up_from_lying/total_count
    g=on_all_fours/total_count
    h=sitting_on_the_ground/total_count
    i=standing_up_from_sitting/total_count
    j=standing_up_from_sitting_on_ground/total_count
    k=lying/total_count
    #print(a,b,c,d,e,f,g,h,i,j,k,total_count)
    factor=0
    # each iteration iterates through the given set and calculates Entropy
    if a!=0:
            factor+=a*math.log2(a)
    elif b!=0:
               factor+=b*math.log2(b)
    elif c!=0:
               factor+=c*math.log2(c)
    elif d!=0:
               factor+=d*math.log2(d)
    elif e!=0:
               factor+=e*math.log2(e)
    elif f!=0:
               factor+=f*math.log2(f)
    elif g!=0:
               factor+=g*math.log2(g)
    elif h!=0:
               factor+=h*math.log2(h)
    elif i!=0:
               factor+=i*math.log2(i)
    elif j!=0:
               factor+=j*math.log2(j)
    elif k!=0:
               factor+=k*math.log2(k)
    entropy = -1 * factor
    #selecting the final entropy based on negative times of given factor
    if a==1:
        ans=1
    elif b==1:
        ans=2
    elif c==1:
        ans=3
    elif d==1:
        ans=4
    elif e==1:
         ans=5
    elif f==1:
        ans=6
    elif g==1:
        ans=7
    elif h==1:
        ans=8
    elif i==1:
        ans=9
    elif j==1:
        ans=10
    elif k==1:
        ans=11

    return entropy, ans

"""#### Defining a function findMaxGain takes 3 arguments data(act), rows, columns and returns maxGain
based on target classification of the activity on the selecting attribute
separation from the training set"""

def findMaxGain(data, rows, columns):
    maxGain = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        """if ans == 1:
            print("Yes")
        else:
            print("No")"""
        return maxGain, retidx, ans

    for jj in columns:
        mydict = {}
        idx = jj
        for ii in rows:
            key = data[ii][idx]
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] = mydict[key] + 1
        gain = entropy

        # print(mydict)
        for key in mydict:
            walking=0
            falling=0
            l_down=0
            s_down=0
            sitting=0
            lying=0
            standing_up_from_lying=0
            on_all_fours=0
            sitting_on_the_ground=0
            standing_up_from_sitting=0
            standing_up_from_sitting_on_ground=0
            total_count=0
            for k in rows:
                if data[k][jj] == key:
                    if data[ii][idx] == 'walking':
                        walking+= 1
                    elif data[ii][idx] == 'falling':
                        falling+=1
                    elif data[ii][idx] == 'lying down':
                        l_down+=1
                    elif data[ii][idx] == 'sitting down':
                        s_down+=1
                    elif data[ii][idx] == 'sitting':
                        sitting+=1
                    elif data[ii][idx] == 'standing up from lying':
                        standing_up_from_lying+=1
                    elif data[ii][idx] == 'on all fours':
                        on_all_fours+=1
                    elif data[ii][idx] == 'sitting on the ground':
                         sitting_on_the_ground+=1
                    elif data[ii][idx] == 'standing up from sitting':
                         standing_up_from_sitting+=1
                    elif data[ii][idx] == 'standing up from sitting on the ground':
                        standing_up_from_sitting_on_ground+=1
                    else:
                        lying+=1
            # print(yes, no)
            # sum of the total activity for the gini index consideration based on gain
            total_count+=walking+falling+l_down+s_down+sitting+standing_up_from_lying+sitting_on_the_ground+standing_up_from_sitting+standing_up_from_sitting_on_ground+lying
            #split each values of Activity classifying based on total_count (DTC)
            a=walking/total_count
            b=falling/total_count
            c=l_down/total_count
            d=s_down/total_count
            e=sitting/total_count
            f=standing_up_from_lying/total_count
            g=on_all_fours/total_count
            h=sitting_on_the_ground/total_count
            i=standing_up_from_sitting/total_count
            j=standing_up_from_sitting_on_ground/total_count
            k=lying/total_count
            # print(x, y)
            # print(a,b,c,d,e,f,g,h,i,j,k,total_count)
            #factor the gain value on each attribute to takes log to determine the max gain for the given set(act)
            factor=0
            if a!=0:
              if a!=0:
               factor+=a*math.log2(a)
            elif b!=0:
               factor+=b*math.log2(b)
            elif c!=0:
               factor+=c*math.log2(c)
            elif d!=0:
                factor+=d*math.log2(d)
            elif e!=0:
                factor+=e*math.log2(e)
            elif f!=0:
                  factor+=f*math.log2(f)
            elif g!=0:
                factor+=g*math.log2(g)
            elif h!=0:
                factor+=h*math.log2(h)
            elif i!=0:
                factor+=i*math.log2(i)
            elif j!=0:
                factor+=j*math.log2(j)
            elif k!=0:
                factor+=k*math.log2(k)
            # gain += (mydict[key] * (a*math.log2(a) + b*math.log2(b)+ c*math.log2(c)+ d*math.log2(d) +e*math.log2(e)+ f*math.log2(f) +g*math.log2(g) +h*math.log2(h)+ i*math.log2(i)+ j*math.log2(j)+k*math.log2(k)))/(rows-1)

            gain+=((mydict[key]*factor)/(Rows))
        # print(gain)
        # conditionality for checking the gain if max than Maxgain the re-assigned
        if gain > maxGain:

            maxGain = gain
            retidx = jj

    return maxGain, retidx, ans

"""#### New function buildTree with attributes data, rows and columns to build the final decision
classifier (considering max gain, entropy)"""

def buildTree(data, rows, columns):

    maxGain, idx, ans = findMaxGain(X, rows, columns)
    #Making Node defining childs of root
    root = Node()
    root.childs = []
    # print(maxGain
    #
    # )
    # selecting the max gain value based on each classification decision on ans value
    if maxGain == 0:
        if ans == 1:
            root.value = 'walking'
        elif ans==2:
            root.value = 'falling'
        elif ans==3:
            root.value='lying down'
        elif ans==4:
            root.value='sitting down'
        elif ans==5:
            root.value='sitting'
        elif ans==6:
            root.value='standing up from lying'
        elif ans==7:
            root.value='on all fours'
        elif ans==8:
            root.value='sitting on the ground'
        elif ans==9:
            root.value='standing up from sitting '
        elif ans==10:
            root.value='standing up from sitting on the ground '
        else:
            root.value='lying'
        return root

    root.value = attribute[idx]
    mydict = {}
    for i in rows:
        key = data[i][idx]
        if key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1

    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx)
    for key in mydict:
        newrows = []
        for i in rows:
            if data[i][idx] == key:
                newrows.append(i)
        # print(newrows)
        #building Tree based on the decison as key parameter
        temp = buildTree(data, newrows, newcolumns)
        temp.decision = key
        root.childs.append(temp)
    return root

#traverse/print the tree with value (key, value) -> decisions
def traverse(root):
    print(root.decision)
    print(root.value)

    n = len(root.childs)
    if n > 0:
      # each iteration w.r.t root taverse child node
        for i in range(0, n):
            traverse(root.childs[i])

# calculate function that does the final call towards the buildTree with defined rows and columns
def calculate():
    row = [i for i in range(0,1200)]
    columns = [i for i in range(1, 6)]
    root = buildTree(X, row, columns)
    root.decision = 'Start'
    traverse(root)

calculate()