import pickle
import os
import numpy as np
from sklearn.svm import LinearSVC


class SimpleClassifier:
    def __init__(self, epoch, cp, tr_size, iteration):
        self.epoch = epoch
        self.cp = cp
        self.num_classes = int(self.cp[4].max() +1)
        self.tr_size = tr_size
        self.clf = LinearSVC(random_state=0, tol=1e-5)
        self.iteration = iteration
        self.whole_score = self.get_whole_score()
        self.pair_score = self.get_two_class_score()

    def get_whole_score(self):
        mean_score = []
        for i in range(self.iteration):
            f_tr_full, f_te_full, y_tr_full, y_te_full = self.get_training_sample(self.tr_size)
            self.clf.fit(f_tr_full, y_tr_full)
            mean_score.append(self.clf.score(f_te_full, y_te_full))
        return sum(mean_score)/len(mean_score)

    def get_two_class_score(self):
        class_list = list(range(self.num_classes))
        all_pair_score = []
        for class_one in class_list:
            if class_one == class_list[-1]:
                break
            pair_score = []
            for class_two in class_list[class_one+1:]:
                mean_score = []
                for i in range(self.iteration):
                    f_tr_full, f_te_full, y_tr_full, y_te_full = self.get_two_class_sample(self.tr_size, [class_one], [class_two])
                    self.clf.fit(f_tr_full, y_tr_full)
                    mean_score.extend(self.clf.score(f_te_full, y_te_full))
                pair_score.append(sum(mean_score)/len(mean_score))
            all_pair_score.append(pair_score)
        return all_pair_score

    def get_training_sample(self, tr_size):
        self.idx_tr_save = []
        self.idx_te_save = []
        self.feat_tr_save = []
        self.feat_te_save = []
        y_tr_save = []
        y_te_save = []
        for i in range(self.num_classes):
            idx_all = np.where(self.cp[4] == i)[0]
            np.random.shuffle(idx_all)
            idx_tr = idx_all[:tr_size]
            idx_te = idx_all[tr_size:]
            feat_tr = self.cp[0][idx_tr]
            feat_te = self.cp[0][idx_te]
            y_tr = np.tile(i, len(idx_tr))
            y_te = np.tile(i, len(idx_te))
            self.idx_tr_save.append(idx_tr)
            self.idx_te_save.append(idx_te)
            self.feat_tr_save.append(feat_tr)
            self.feat_te_save.append(feat_te)
            y_tr_save.extend(y_tr)
            y_te_save.extend(y_te)
        f_tr_save = np.vstack(self.feat_tr_save)
        f_te_save = np.vstack(self.feat_te_save)
        return f_tr_save, f_te_save, y_tr_save, y_te_save

    def get_two_class_sample(self, tr_size, class_one, class_two):
        self.idx_tr_one_save = []
        self.idx_te_one_save = []
        self.feat_tr_one_save = []
        self.feat_te_one_save = []
        y_tr_one_save = []
        y_te_one_save = []
        for i in class_one:
            idx_all = np.where(self.cp[4] == i)[0]
            np.random.shuffle(idx_all)
            idx_tr_one = idx_all[:tr_size]
            idx_te_one = idx_all[tr_size:]
            feat_tr_one = self.cp[0][idx_tr_one]
            feat_te_one = self.cp[0][idx_te_one]
            y_tr_one = np.tile(0, len(idx_tr_one))
            y_te_one = np.tile(0, len(idx_te_one))
            self.idx_tr_one_save.append(idx_tr_one)
            self.idx_te_one_save.append(idx_te_one)
            self.feat_tr_one_save.append(feat_tr_one)
            self.feat_te_one_save.append(feat_te_one)
            y_tr_one_save.extend(y_tr_one)
            y_te_one_save.extend(y_te_one)
        f_tr_one_save = np.vstack(self.feat_tr_one_save)
        f_te_one_save = np.vstack(self.feat_te_one_save)

        self.idx_tr_two_save = []
        self.idx_te_two_save = []
        self.feat_tr_two_save = []
        self.feat_te_two_save = []
        y_tr_two_save = []
        y_te_two_save = []
        for i in class_two:
            idx_all = np.where(self.cp[4] == i)[0]
            np.random.shuffle(idx_all)
            idx_tr_two = idx_all[:tr_size]
            idx_te_two = idx_all[tr_size:]
            feat_tr_two = self.cp[0][idx_tr_two]
            feat_te_two = self.cp[0][idx_te_two]
            y_tr_two = np.tile(1, len(idx_tr_two))
            y_te_two = np.tile(1, len(idx_te_two))
            self.idx_tr_two_save.append(idx_tr_two)
            self.idx_te_two_save.append(idx_te_two)
            self.feat_tr_two_save.append(feat_tr_two)
            self.feat_te_two_save.append(feat_te_two)
            y_tr_two_save.extend(y_tr_two)
            y_te_two_save.extend(y_te_two)
        f_tr_two_save = np.vstack(self.feat_tr_two_save)
        f_te_two_save = np.vstack(self.feat_te_two_save)
        f_tr_save = np.concatenate([f_tr_one_save, f_tr_two_save])
        f_te_save = np.concatenate([f_te_one_save, f_te_two_save])
        y_tr_save = np.concatenate([y_tr_one_save, y_tr_two_save])
        y_te_save = np.concatenate([y_te_one_save, y_te_two_save])
        return f_tr_save, f_te_save, y_tr_save, y_te_save


class FeatureLoad:
    def __init__(self, epoch):
        self.home_path = '/Users/changkyu/Documents/GitHub/deepcluster_analysis'
        self.dataset_path = '200428_MLSP'
        self.epoch = epoch
        self.cp = self.cp_load()
        self.num_classes = int(self.cp[4].max() +1)

    def cp_load(self):
        with open(os.path.join(self.home_path, self.dataset_path, 'cp_epoch_%d.pickle' % self.epoch), 'rb') as f:
            cp_epoch = pickle.load(f)
        return cp_epoch

    def get_training_sample(self, tr_size):
        self.idx_tr_save = []
        self.idx_te_save = []
        self.feat_tr_save = []
        self.feat_te_save = []
        y_tr_save = []
        y_te_save = []
        for i in range(self.num_classes):
            idx_all = np.where(self.cp[4] == i)[0]
            np.random.shuffle(idx_all)
            idx_tr = idx_all[:tr_size]
            idx_te = idx_all[tr_size:]
            feat_tr = self.cp[0][idx_tr]
            feat_te = self.cp[0][idx_te]
            y_tr = np.tile(i, len(idx_tr))
            y_te = np.tile(i, len(idx_te))
            self.idx_tr_save.append(idx_tr)
            self.idx_te_save.append(idx_te)
            self.feat_tr_save.append(feat_tr)
            self.feat_te_save.append(feat_te)
            y_tr_save.extend(y_tr)
            y_te_save.extend(y_te)
        f_tr_save = np.vstack(self.feat_tr_save)
        f_te_save = np.vstack(self.feat_te_save)
        # np.arange()
        return f_tr_save, f_te_save, y_tr_save, y_te_save

    def get_two_class_sample(self, tr_size, class_one, class_two):
        self.idx_tr_one_save = []
        self.idx_te_one_save = []
        self.feat_tr_one_save = []
        self.feat_te_one_save = []
        y_tr_one_save = []
        y_te_one_save = []
        for i in class_one:
            idx_all = np.where(self.cp[4] == i)[0]
            np.random.shuffle(idx_all)
            idx_tr_one = idx_all[:tr_size]
            idx_te_one = idx_all[tr_size:]
            feat_tr_one = self.cp[0][idx_tr_one]
            feat_te_one = self.cp[0][idx_te_one]
            y_tr_one = np.tile(0, len(idx_tr_one))
            y_te_one = np.tile(0, len(idx_te_one))
            self.idx_tr_one_save.append(idx_tr_one)
            self.idx_te_one_save.append(idx_te_one)
            self.feat_tr_one_save.append(feat_tr_one)
            self.feat_te_one_save.append(feat_te_one)
            y_tr_one_save.extend(y_tr_one)
            y_te_one_save.extend(y_te_one)
        f_tr_one_save = np.vstack(self.feat_tr_one_save)
        f_te_one_save = np.vstack(self.feat_te_one_save)

        self.idx_tr_two_save = []
        self.idx_te_two_save = []
        self.feat_tr_two_save = []
        self.feat_te_two_save = []
        y_tr_two_save = []
        y_te_two_save = []
        for i in class_two:
            idx_all = np.where(self.cp[4] == i)[0]
            np.random.shuffle(idx_all)
            idx_tr_two = idx_all[:tr_size]
            idx_te_two = idx_all[tr_size:]
            feat_tr_two = self.cp[0][idx_tr_two]
            feat_te_two = self.cp[0][idx_te_two]
            y_tr_two = np.tile(1, len(idx_tr_two))
            y_te_two = np.tile(1, len(idx_te_two))
            self.idx_tr_two_save.append(idx_tr_two)
            self.idx_te_two_save.append(idx_te_two)
            self.feat_tr_two_save.append(feat_tr_two)
            self.feat_te_two_save.append(feat_te_two)
            y_tr_two_save.extend(y_tr_two)
            y_te_two_save.extend(y_te_two)
        f_tr_two_save = np.vstack(self.feat_tr_two_save)
        f_te_two_save = np.vstack(self.feat_te_two_save)
        f_tr_save = np.concatenate([f_tr_one_save, f_tr_two_save])
        f_te_save = np.concatenate([f_te_one_save, f_te_two_save])
        y_tr_save = np.concatenate([y_tr_one_save, y_tr_two_save])
        y_te_save = np.concatenate([y_te_one_save, y_te_two_save])
        return f_tr_save, f_te_save, y_tr_save, y_te_save

# epoch = 99
# f = FeatureLoad(epoch)
# # f_tr, f_te, y_tr, y_te= f.get_training_sample(tr_size=5)
#
# class_one = [0]
# class_two = [1]
# f_tr, f_te, y_tr, y_te = \
#     f.get_two_class_sample(tr_size=5, class_one=class_one, class_two=class_two)
#
#
