import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
df = pd.read_csv('spam dataset.csv')

df.drop_duplicates(inplace=True)
print(df.shape)
df['Sentiment']=df['Sentiment'].replace(['positive','neutral','negative'],['not spam','may be spam','spam'])
print(df.head(5))
p_n=df['product_name']
St=df['Sentiment']
(p_n_train,p_n_test,St_train,St_test)=train_test_split(p_n,St,test_size=0.2)
cv=CountVectorizer(stop_words='english')
feature=cv.fit_transform(p_n_train)

##creting the model
model=MultinomialNB()
model.fit(feature,St_train)

##test our model
featrures_test = cv.transform(p_n_test)
print(model.score(featrures_test,St_test))

##predict the output
message = cv.transform(['congratulation ']).toarray()
result= model.predict(message)
print(result)



