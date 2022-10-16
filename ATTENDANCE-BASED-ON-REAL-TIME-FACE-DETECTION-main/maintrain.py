# import libraries
import pickle
from sklearn.svm import SVC

print("[INFO] loading face embeddings...")
data1 = pickle.loads(open("output/encode.pickle","rb").read())
print("[INFO] training model...")
recogniz = SVC(C=1.0, kernel='rbf',decision_function_shape='ovr')
recogniz.fit(data1["encodings"], data1["namess"])
f = open("output/recognizerE.pickle", "wb")
f.write(pickle.dumps(recogniz))
f.close()



