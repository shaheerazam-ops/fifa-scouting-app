"""
import pandas as pd
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix ,f1_score ,recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Fifa_23_Players_Data.csv', low_memory=False)
print(df.shape)
print(df["Best Position"].value_counts())
df["Attack_Score"] = df["Shooting Total"] + df["Pace Total"] + df["Dribbling Total"]
df["Midfield_Score"] = df["Passing Total"] + df["Dribbling Total"]
df["Defense_Score"] = df["Defending Total"] + df["Physicality Total"]

df["Attack_vs_Defense"] = df["Attack_Score"] / (df["Defense_Score"] + 1)
df["Dribble_vs_Physical"] = df["Dribbling Total"] / (df["Physicality Total"] + 1)
df["Pass_vs_Shoot"] = df["Passing Total"] / (df["Shooting Total"] + 1)

features = [
    "Age",
    "Overall",
    "Potential",
    "Pace Total",
    "Shooting Total",
    "Passing Total",
    "Dribbling Total",
    "Defending Total",
    "Physicality Total",
    "Attack_Score",
    "Midfield_Score",
    "Defense_Score",
    "Attack_vs_Defense",
    "Dribble_vs_Physical",
    "Pass_vs_Shoot"
]

x = df[features]
y = df["Best Position"]


x_train , x_test , y_train , y_test = train_test_split(
    x , y ,test_size = 0.2 , random_state  = 42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=5000 , class_weight = 'balanced'))
])


model.fit(x_train,y_train)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test , y_pred)
print(acc)
print(classification_report(y_test,y_pred))
print('cv' , cross_val_score(model,x,y,cv=5).mean())

pipeline = Pipeline([
    ('scaler' , StandardScaler()),
    ('rfc' , RandomForestClassifier(n_estimators=150 ,random_state = 42))
])
pipeline.fit(x_train,y_train)
y_pred_pipline = pipeline.predict(x_test)
score = cross_val_score(pipeline , x , y , cv= 5 )
print('cv',score)
print('mean cv score',score.mean())

rf_model = pipeline.named_steps["rfc"]

importance = pd.Series(rf_model.feature_importances_, index=x.columns)
print(importance.sort_values(ascending=False))
"""
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

texts = [
# POSITIVE
"ronaldo is the greatest player",
"messi played an amazing match",
"neymar was brilliant today",
"mbappe is unstoppable",
"what a fantastic goal",
"this team played beautifully",
"incredible performance from the striker",
"he scored a stunning goal",
"what a world class assist",
"defense was rock solid",
"goalkeeper saved everything",
"this was a dominant win",
"great teamwork and passing",
"what a legendary performance",
"he controlled the midfield perfectly",
"the attack was very sharp",
"excellent dribbling skills",
"that was a perfect finish",
"what a comeback by the team",
"they played with great energy",

"ronaldo showed elite mentality",
"messi vision is unbelievable",
"great pressing and control",
"amazing counter attack",
"this squad is very strong",
"beautiful football by the team",
"perfect execution of tactics",
"he is a game changer",
"brilliant playmaking",
"top class performance",

"that goal was insane",
"he is very talented",
"great chemistry between players",
"amazing speed and control",
"they deserved the win",
"solid defensive performance",
"great attacking movement",
"perfect positioning",
"fantastic team effort",
"he played like a superstar",

# NEGATIVE
"that was a terrible match",
"he played very badly",
"worst performance ever",
"awful defending",
"goalkeeper made huge mistakes",
"team had no coordination",
"they played like amateurs",
"very poor passing",
"he missed an easy chance",
"terrible finishing",

"this team is weak",
"they got completely destroyed",
"bad decision making",
"very disappointing result",
"they lacked energy",
"defense was horrible",
"attack was useless",
"midfield was invisible",
"worst game of the season",
"he was completely outplayed",

"that was embarrassing",
"they made too many mistakes",
"no creativity in attack",
"very sloppy performance",
"poor ball control",
"they failed badly",
"he lost the ball too often",
"team showed no effort",
"they were dominated",
"this was a disaster",

"terrible tactics by the coach",
"no teamwork at all",
"very frustrating to watch",
"players looked lost",
"bad positioning",
"he was too slow",
"no finishing ability",
"they wasted chances",
"defense collapsed",
"completely unacceptable performance",

# MIXED / TRICKY (important for learning)
"messi played well but missed chances",
"ronaldo scored but team lost",
"good attack but poor defense",
"great first half terrible second half",
"he was good but inconsistent",
"team started strong but collapsed",
"decent performance overall",
"not bad but not great",
"average gameplay today",
"some good moments but many mistakes",
"it was okay",
"average performance",
"nothing special",
"decent but not impressive",
"could have been better",
"it was fine",
"average performance",
"not impressive but okay",
"decent game overall",
"nothing special happened"
]

labels = [
# POSITIVE (40)
"positive","positive","positive","positive","positive","positive","positive","positive","positive","positive",
"positive","positive","positive","positive","positive","positive","positive","positive","positive","positive",
"positive","positive","positive","positive","positive","positive","positive","positive","positive","positive",
"positive","positive","positive","positive","positive","positive","positive","positive","positive","positive",

# NEGATIVE (40)
"negative","negative","negative","negative","negative","negative","negative","negative","negative","negative",
"negative","negative","negative","negative","negative","negative","negative","negative","negative","negative",
"negative","negative","negative","negative","negative","negative","negative","negative","negative","negative",
"negative","negative","negative","negative","negative","negative","negative","negative","negative","negative",

# MIXED (10 → label as neutral OR choose one)
"neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral","neutral"
]

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

lr_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english" , ngram_range = (1,2) , max_features =1000)),
    ("clf", LogisticRegression(max_iter=5000,class_weight = "balanced")),
    #('NB',MultinomialNB (max_iter=5000,class_weight = "balanced"))
])

nb_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english" , ngram_range = (1,2) , max_features =1000)),
    #("clf", LogisticRegression(max_iter=5000,class_weight = "balanced")),
    ('NB',MultinomialNB ())
])

scores = cross_val_score(lr_model ,X_train , y_train,cv=5)
print('cv scores :' , scores)
print('mean :' , scores.mean())

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_test)

print(classification_report(y_test, y_pred))

print(lr_model.predict(["messi played terrible"]))
print(lr_model.predict(["ronaldo is the best"]))


nb_scores = cross_val_score(nb_model ,X_train , y_train,cv=5)
print('cv scores :' , nb_scores)
print('mean :' , nb_scores.mean())

nb_model.fit(X_train, y_train)

y2_pred = nb_model.predict(X_test)

print(classification_report(y_test, y_pred))

print(nb_model.predict(["messi played terrible"]))
print(nb_model.predict(["ronaldo is the best"]))
"""
"""
from gensim.models import Word2Vec

sentences = [
    ["messi", "scored", "goal"],
    ["ronaldo", "scored", "goal"],
    ["neymar", "made", "assist"],
    ["team", "played", "well"]
]


model = Word2Vec(sentences, vector_size=50, window=2, min_count=1)

print(model.wv["messi"])
print(model.wv.most_similar("messi"))
model.wv.similarity("messi", "ronaldo")
"""
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 

model = SentenceTransformer('all-MiniLM-L6-v2')

comments = [
     "ronaldo scored a hattrick",
    "messi gave an amazing assist",
    "this match was boring",
    "defense was terrible today",
    "what a fantastic goal",
    "team played very badly"

]

embeddings = model.encode(comments)

def search(query):

    query_emb = model.encode([query])
    sim = cosine_similarity(query_emb , embeddings)[0]

    best_index = sim.argmax()
    return comments[best_index] , sim ,query_emb

print(embeddings)
print(search("great goal"))
print(search('bad defense'))
"""