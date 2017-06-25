from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import mmfeat.space



def main():
    ling_space = mmfeat.space.Space('wiki.en.json')
    print("Linguistics space loaded")
    vis_space = mmfeat.space.Space('esp_cnn.pkl')
    print("Visual space loaded")
    mm_space = mmfeat.space.mmspace.MMSpace(ling_space, vis_space)
    print("Multimodal space loaded")
    # Set the parameters by cross-validation
    tuned_parameters = [{'alpha': np.linspace(0, 1, 10)}]

    print("# Tuning hyper-parameters: alpha")
    print()

    estim = mmfeat.space.mmspace.MMEstimator(mm_space)
    estim.loadDataset(datasetLocation='/home/geopar/projects/multilearn/mmfeat/datasets')
    clf = GridSearchCV(estim, tuned_parameters, cv=5)
    X_train = [list(x) for x in estim.actual_values.keys()]
    y_train = list(estim.actual_values.values())
    print("Fitting GridSearchCV estimator")
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
        print()

#    print("Detailed classification report:")
#    print()
#    print("The model is trained on the full development set.")
#    print("The scores are computed on the full evaluation set.")
#    print()
#    y_true, y_pred = y_test, clf.predict(X_test)
#    print(classification_report(y_true, y_pred))
#    print()

if __name__ == '__main__':
    main()
