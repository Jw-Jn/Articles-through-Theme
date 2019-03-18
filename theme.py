import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

n_components = 2 
n_topics = 5 

# read data
df = pd.read_csv("articles.csv")
content = df['content']
themes = []

# get themes
tfidfVectorizer = TfidfVectorizer(stop_words='english')

for i in range(content.shape[0]):

    tf = tfidfVectorizer.fit_transform([content[i]])

    lda = LatentDirichletAllocation(n_components=n_components, learning_method='online', random_state=0)
    lda.fit(tf)

    feature_names = tfidfVectorizer.get_feature_names()

    temp = []
    for _, topic in enumerate(lda.components_):
        temp.append([feature_names[i] for i in topic.argsort()[:-n_topics - 1:-1]])

    themes.append(temp)

    # print('article ', i+1)
    # print(temp)
    # print()

# define a theme, get articles
print('input a theme')
theme = input()
for article_idx in range(len(themes)):
    for article_theme in themes[article_idx]:
        if theme in article_theme:
            print('article:', article_idx+1, df['title'][article_idx])
            break