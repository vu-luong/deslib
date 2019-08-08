import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from deslib.des import METADES
from sklearn.metrics import precision_recall_fscore_support
import os


def write_file(array, folder, filename):
    array_mean = np.mean(array)
    array_var = np.var(array)
    np.savetxt(folder + "/" + filename, array, delimiter=',', fmt='%0.6e')
    f = open(folder + "/" + filename, "a")
    f.write("----------\n")
    f.write("Mean:\n")
    f.write("{0:6E}\n".format(array_mean))
    f.write("Variance:\n")
    f.write("{0:6E}".format(array_var))
    f.close()


file_list = ["GM4", "Skin_NonSkin", "abalone", "appendicitis", "artificial1",
             "australian", "balance", "banana", "biodeg", "blood",
             "breast-cancer", "BreastTissue", "bupa", "chess-krvk", "cleveland",
             "conn-bench-vowel", "contraceptive", "dermatology", "fertility", "haberman",
             "hayes-roth", "heart", "hepatitis", "hill_valley", "iris", "isolet",
             "led7digit_new", "letter", "libras", "madelon", "magic", "mammographic",
             "marketing", "monk-2_new", "multiple-features", "musk1", "musk2", "newthyroid",
             "page-blocks", "penbased_new_fix", "phoneme", "pima", "plant_margin",
             "plant_shape", "plant_texture", "ring1", "satimage", "shuttle", "sonar",
             "spambase", "tae", "texture-fix", "tic-tac-toe", "titanic_new", "twonorm1",
             "vehicle", "vertebral_3C", "waveform_w_noise", "waveform_wo_noise", "wdbc",
             "wine", "wine_red", "wine_white", "yeast"]

file_list = ["abalone"]

data_folder = "/Users/AnhVu/Study/Machine_learning/Data/convert/csv"
cv_folder = "/Users/AnhVu/Study/Machine_learning/Data/convert/cv"

# Parameters
n_folds = 10
n_iters = 1
validation_rate = 0.3

rng = np.random.RandomState(42)

for file_name in file_list:
    print(file_name)

    D = np.loadtxt("{}/{}.csv".format(data_folder, file_name), delimiter=',')
    cv = sio.loadmat("{}/cv_{}.mat".format(cv_folder, file_name))['cv']

    print("D.shape = {}".format(D.shape))

    n_instances = D.shape[0]
    n_features = D.shape[1] - 1

    all_ids = np.array(range(D.shape[0]))

    errors = np.zeros(n_iters * n_folds, )
    precisions_macro = np.zeros(n_iters * n_folds, )
    recalls_macro = np.zeros(n_iters * n_folds, )
    f1s_macro = np.zeros(n_iters * n_folds, )

    precisions_micro = np.zeros(n_iters * n_folds, )
    recalls_micro = np.zeros(n_iters * n_folds, )
    f1s_micro = np.zeros(n_iters * n_folds, )

    for i_iter in range(n_iters):
        base_loop = i_iter * n_folds
        for i_fold in range(n_folds):
            current_loop = base_loop + i_fold

            # subtract one since index in python starts from 0 (matlab from 0)
            test_ids = cv[0, current_loop][:, 0] - 1
            train_ids = np.setdiff1d(all_ids, test_ids)

            X_train = D[train_ids, :-1]
            Y_train = D[train_ids, -1]

            X_test = D[test_ids, :-1]
            Y_test = D[test_ids, -1]

            X_train, X_dev, Y_train, Y_dev = train_test_split(X_train, Y_train,
                                                              test_size=0.3,
                                                              random_state=rng)

            # Base Classifiers
            model_knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, Y_train)
            model_nb = GaussianNB().fit(X_train, Y_train)
            model_lda = LinearDiscriminantAnalysis().fit(X_train, Y_train)

            pool_classifiers = [model_lda, model_nb, model_knn]

            metades = METADES(pool_classifiers)

            metades.fit(X_dev, Y_dev)

            accuracy = metades.score(X_test, Y_test)
            Y_pred = metades.predict(X_test)
            support_macro = precision_recall_fscore_support(Y_test, Y_pred, average='macro')
            support_micro = precision_recall_fscore_support(Y_test, Y_pred, average='micro')

            print('Classification accuracy of META-DES: ', accuracy)

            # Save results
            errors[current_loop] = 1 - accuracy

            precisions_macro[current_loop] = support_macro[0]
            recalls_macro[current_loop] = support_macro[1]
            f1s_macro[current_loop] = support_macro[2]

            precisions_micro[current_loop] = support_micro[0]
            recalls_micro[current_loop] = support_micro[1]
            f1s_micro[current_loop] = support_micro[2]

    result_folder = "result/{}".format(file_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    write_file(errors, result_folder, "err.dat")
    write_file(precisions_macro, result_folder, "precision_macro.dat")
    write_file(recalls_macro, result_folder, "recall_macro.dat")
    write_file(f1s_macro, result_folder, "f1_macro.dat")

    write_file(precisions_micro, result_folder, "precision_micro.dat")
    write_file(recalls_micro, result_folder, "recall_micro.dat")
    write_file(f1s_micro, result_folder, "f1_micro.dat")
