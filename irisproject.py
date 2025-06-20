# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# %%
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length","sepal_width", "petal_length","petal_width", "class"]
iris_data = pd.read_csv( url , names= column_names)

# %%
iris_data.head

# %%
iris_data.head(5)

# %%
iris_data.iloc[50:100]

# %%
iris_data.describe()

# %%
sns.pairplot(iris_data, hue="class")
plt.show()

# %%
x = iris_data.drop("class" , axis = 1)
x

# %%
y = iris_data["class"]
y

# %%
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= 0.3 , random_state=42)

# %%
knn= KNeighborsClassifier(n_neighbors= 3)
knn.fit(x_train, y_train)

# %%
y_pred = knn.predict(x_test)

# %%
print("Accuracy:", accuracy_score(y_test, y_pred))


