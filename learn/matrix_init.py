import numpy as np
from scipy.stats import truncnorm


#################
################# MATRIX INITIALIZATION
#################

class matrix_init:

    def truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

#     X = truncated_normal(mean=0, sd=0.4, low=-0.5, upp=0.5)
#     truncated_normal_dataset = X.rvs(10000)

#     plt.hist(truncated_normal_dataset)

