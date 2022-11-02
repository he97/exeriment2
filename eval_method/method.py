import numpy as np
from sklearn import metrics


class Eval_method:
    def __init__(self, seed, class_num):
        self.seed = seed
        self.class_num = class_num
        self.seed_len = len(seed)
        self.this_seed_idx = -1
        self.acc = np.zeros([self.seed_len, 1])
        self.A = np.zeros([self.seed_len, self.class_num])
        self.K = np.zeros([self.seed_len, 1])

    def set_seed(self, this_seed):
        if this_seed not in self.seed:
            raise Exception('this seed is invalid')
        else:
            self.this_seed_idx = self.seed.index(this_seed)

    def set_value(self, acc, val_all, val_pred_all):
        if self.acc[self.this_seed_idx] == -1:
            raise IndexError("not set seed index")
        if self.acc[self.this_seed_idx] < acc:
            self.acc[self.this_seed_idx] = acc
            C = metrics.confusion_matrix(val_all, val_pred_all)
            self.A[self.this_seed_idx, :] = np.diag(C) / np.sum(C, 1, dtype=np.float)
            self.K[self.this_seed_idx] = metrics.cohen_kappa_score(val_all, val_pred_all)

    def get_OA_AA_KAPPA(self, logger):
        AA = np.mean(self.A, 1)
        AAMean = np.mean(AA, 0)
        AAStd = np.std(AA)
        AMean = np.mean(self.A, 0)
        AStd = np.std(self.A, 0)
        OAMean = np.mean(self.acc)
        OAStd = np.std(self.acc)
        kMean = np.mean(self.K)
        kStd = np.std(self.K)
        for iDataSet in range(self.seed_len):
            logger.info('Run: {}\tval_Accuracy: ({:.2f}%)\t'.format(iDataSet, self.acc[iDataSet][0]))
        logger.info("average OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
        logger.info("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
        logger.info("average kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
        logger.info("accuracy for each class: ")
        for i in range(self.class_num):
            logger.info(
                "Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))
