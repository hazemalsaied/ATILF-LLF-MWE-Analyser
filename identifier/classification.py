import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

# vecData = 'vecs_data.pkl'
# clfData = 'cls_data.pkl'
clfData = '../Corpora/Models/cls_data.pkl'
vecData = '../Corpora/Models/vecs_data.pkl'


class SVMClf:
    def __init__(self, labels, data, load=False, save=False):
        if load:
            with open(clfData, 'rb') as input:
                self.classifier = pickle.load(input)
            with open(vecData, 'rb') as input:
                self.verctorizer = pickle.load(input)
            return
        self.verctorizer = DictVectorizer()
        featureVec = self.verctorizer.fit_transform(data)
        self.classifier = OutputCodeClassifier(LinearSVC(random_state=0), code_size=2, random_state=0)
        # self.classifier = LogisticRegression( solver='sag')
        self.classifier.fit(featureVec, labels)
        if save:
            with open(clfData, 'wb') as output:
                pickle.dump(self.classifier, output, pickle.HIGHEST_PROTOCOL)
            with open(vecData, 'wb') as output:
                pickle.dump(self.verctorizer, output, pickle.HIGHEST_PROTOCOL)
