import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from Modules import K_Nearest_Neighbour as KNN


data = pd.read_csv("../Dataset/posturalDatset.csv")

x = data.iloc[:, :-1]
x = x.iloc[:, 2:]
x.drop(['Date'], axis=1, inplace=True)

y = data.iloc[:, -1]

y = y.to_frame()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)



mapping = {"walking": 1, "sitting down": 2, "falling": 3, "lying down": 4, "lying": 5, "sitting": 6,
           "standing up from lying": 7, "on all fours": 8, "sitting on the ground": 9, "standing up from sitting": 10,
           "standing up from sitting on the ground": 11}

y_train['Activitynum'] = data['Activity'].map(mapping)
y_test['Activitynum'] = data['Activity'].map(mapping)

x_train['BodyAccMagMean'] = (x_train['x'] + x_train['y'] + x_train['z']) / 3
x_test['BodyAccMagMean'] = (x_test['x'] + x_test['y'] + x_test['z']) / 3

y_train['BodyAccMagMean'] = x_train['BodyAccMagMean']
print(y_train)
df1 = y_train[y_train['Activity'] == 'walking']
df2 = y_train[y_train['Activity'] == 'sitting down']
df3 = y_train[y_train['Activity'] == 'falling']
df4 = y_train[y_train['Activity'] == 'lying down']
df5 = y_train[y_train['Activity'] == 'lying']
df6 = y_train[y_train['Activity'] == 'sitting']
df7 = y_train[y_train['Activity'] == 'standing up from lying']
df8 = y_train[y_train['Activity'] == 'on all fours']
df9 = y_train[y_train['Activity'] == 'sitting on the ground']
df10 = y_train[y_train['Activity'] == 'standing up from sitting']
df11 = y_train[y_train['Activity'] == 'standing up from sitting on the ground']


plt.figure(figsize=(16, 8))
sns.distplot(df1['BodyAccMagMean'], hist=False, rug=False)
sns.distplot(df2['BodyAccMagMean'], hist=False, rug=False)
sns.distplot(df3['BodyAccMagMean'], hist=False, rug=False)
sns.distplot(df4['BodyAccMagMean'], hist=False, rug=False)
sns.distplot(df5['BodyAccMagMean'], hist=False, rug=False)
sns.distplot(df6['BodyAccMagMean'], hist=False, rug=False, label='sitting')
sns.distplot(df7['BodyAccMagMean'], hist=False, rug=False, label='standing up from lying')
sns.distplot(df8['BodyAccMagMean'], hist=False, rug=False, label='on all fours')
sns.distplot(df9['BodyAccMagMean'], hist=False, rug=False, label='sitting on the ground')
sns.distplot(df10['BodyAccMagMean'], hist=False, rug=False, label='standing up from sitting')
sns.distplot(df11['BodyAccMagMean'], hist=False, rug=False, label='standing up from sitting on the ground')
plt.legend(labels=["walking", "sitting down", "falling", "lying down", "lying", "sitting", "standing up from lying",
                   "on all fours", "sitting on the ground", "standing up from sitting",
                   "standing up from sitting on the ground"])
plt.show()

print(y_train.columns)
print(type(y_train))
plt.figure(figsize=(8, 8))
sns.boxplot(x='Activity', y='BodyAccMagMean', data=y_train)
plt.xticks(rotation=90)
plt.show()
y_train.drop(['BodyAccMagMean'], axis=1, inplace=True)

y_train.drop(['Activity'], axis=1, inplace=True)
y_test.drop(['Activity'], axis=1, inplace=True)


x_train = x_train.iloc[:, -1]
x_train = x_train.to_frame()
x_test = x_test.iloc[:, -1]
x_test = x_test.to_frame()

KNN.KNNClassifier.train_model(x_train, x_test)

pred = KNN.KNNClassifier.classifier(x_train, y_train, x_test, y_test)

pred = pd.DataFrame(pred, columns=['Activitynum'])