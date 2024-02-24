from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier

def get_model():
    clf = MultinomialNB(alpha=0.0225)
    sgd_model = SGDClassifier(max_iter=27000, tol=1e-4, loss="modified_huber", random_state=6743)

    weights = [0.43, 0.57]
 
    ensemble = VotingClassifier(estimators=[('mnb',clf),
                                            ('sgd', sgd_model),
                                           ],
                                weights=weights, voting='soft', n_jobs=-1)
    return ensemble
