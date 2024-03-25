import random
import numpy as np
from pyBKT.util import dirrnd

def random_model_uni(num_resources=None, num_subparts=None, trans_prior=None, given_notknow_prior=None, given_know_prior=None, pi_0_prior=None, cognitive_label=None):
    epsilon = 0.01  # 定义一个小的正数，例如 1% 或者 0.01
    if num_resources is None: num_resources = 1
    if num_subparts is None: num_subparts = 1

    if trans_prior is None:
        trans_prior = np.tile(np.transpose([[20, 4], [1, 20]]), (num_resources, 1)).reshape((num_resources, 2, 2))
    if given_notknow_prior is None:
        given_notknow_prior = np.tile([[5], [0.5]], (1, num_subparts))
    if given_know_prior is None:
        given_know_prior = np.tile([[0.5], [5]], (1, num_subparts))
    if pi_0_prior is None:
        pi_0_prior = np.array([[100], [1]])

    As = dirrnd.dirrnd(trans_prior)
    given_notknow = dirrnd.dirrnd(given_notknow_prior)
    given_know = dirrnd.dirrnd(given_know_prior)
    #emissions = np.dstack((given_notknow.reshape((num_subparts, 2, 1)), given_know.reshape((num_subparts, 2, 1))))
    emissions = np.stack((np.transpose(given_notknow.reshape((2, num_subparts))), np.transpose(given_know.reshape((2, num_subparts)))), axis=1)
    pi_0 = dirrnd.dirrnd(pi_0_prior)

    modelstruct = {}
    modelstruct['prior'] = random.random()
    As[:, 1, 0] = np.random.rand(num_resources) * 0.40
    As[:, 1, 1] = 1 - As[:, 1, 0]
    As[:, 0, 1] = 0
    As[:, 0, 0] = 1
    modelstruct['learns'] = As[:, 1, 0]
    modelstruct['forgets'] = As[:, 0, 1]

    if cognitive_label is None:
        given_notknow[1, :] = np.random.rand(num_subparts) * 0.40
        given_know[0, :] = np.random.rand(num_subparts) * 0.30
    else:
        cognitive_labels = np.array(list(cognitive_label.values()))
        if np.any(cognitive_labels < 1) or np.any(cognitive_labels > 6):
            raise ValueError("All cognitive levels must be between 1 and 6")

        # 为每个题目计算 guess 和 slip 参数
        for i in range(num_subparts):
            # 计算 guess_factor 和 slip_factor
            guess_factor = np.maximum(0.01, np.minimum(0.4, 0.4 - ((cognitive_labels[i] - 1) / 5) * 0.4))
            slip_factor = np.maximum(0.01, np.minimum(0.3, 0.1 + ((cognitive_labels[i] - 1) / 5) * 0.2))

            # 将计算出的因子值赋给 given_notknow 和 given_know 的相应位置
            given_notknow[1, i] = guess_factor
            given_know[0, i] = slip_factor

    modelstruct['guesses'] = given_notknow[1, :]
    modelstruct['slips'] = given_know[0, :]

    modelstruct['As'] = As
    modelstruct['emissions'] = emissions
    modelstruct['pi_0'] = pi_0

    return(modelstruct)