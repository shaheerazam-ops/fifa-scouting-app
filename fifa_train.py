import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_absolute_error , r2_score 
from sklearn.preprocessing import StandardScaler
import pickle 
from sklearn.cluster import KMeans

# ---------------  LOAD DATA ------------------
df = pd.read_csv('Fifa_23_Players_Data.csv', low_memory = False)

df.columns = df.columns.str.strip()
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst row:", df.iloc[0])
print("\nMissing values:\n", df.isnull().sum().sort_values(ascending = False).head(20))

# ---------------- prep data --------

features = [
    'Age',
    'Overall',
    'Potential',
    'Wage(in Euro)',
    'Pace Total',
    'Shooting Total',
    'Passing Total',
    'Dribbling Total',
    'Defending Total',
    'Physicality Total'
]

x=df[features]
y = df['Value(in Euro)']

# ------------train data ---------------

x_train , x_test , y_train , y_test = train_test_split(
    x , y ,test_size=0.2,random_state=42
)

#---------- model pipline --------------

model = Pipeline([
    ('scaler' , StandardScaler()),
    ('rfr' , RandomForestRegressor(n_estimators=100 , random_state = 42))

])

# ------------ cross validation 

scores = cross_val_score(model,x_train,y_train,cv=5,scoring = 'r2')

print("cv ", scores)
print('mean cv score' , scores.mean())

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("\n model evalution")
print("MEA :" , mean_absolute_error(y_test,y_pred))
print("R2 score :" , r2_score(y_test,y_pred))

#------------ save model --------------------

pickle.dump(model,open("fifa_model.pkl",'wb'))
pickle.dump(x.columns.tolist(), open("fifa_features.pkl" , 'wb'))

print("\n model saved succesfully")

#------------------ k means model training -------------

cluster_features = [
    'Pace Total',
    'Shooting Total',
    'Passing Total',
    'Dribbling Total',
    'Defending Total',
    'Physicality Total'
]

kmeans = KMeans(n_clusters = 5 , random_state = 42 , n_init = 10)
df["cluster"] = kmeans.fit_predict(df[cluster_features])
print(df["cluster"].value_counts())

# ------------- saving model ---------------

pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))
df.to_csv("fifa_with_clusters.csv", index=False)


similarity_features = [
    'Pace Total',
    'Shooting Total',
    'Passing Total',
    'Dribbling Total',
    'Defending Total',
    'Physicality Total'
]

x_sim = df[similarity_features]
k_scaler = StandardScaler()
x_scaled = k_scaler.fit_transform(x_sim)

knn = NearestNeighbors(n_neighbors = 6 , metric = 'euclidean')
knn.fit(x_scaled)

pickle.dump(knn , open("knn_model.pkl","wb"))
pickle.dump(similarity_features , open("knn_features.pkl","wb"))
pickle.dump(k_scaler , open("knn_scaler.pkl","wb"))
print("knn model saved")
