import numpy as np
from sklearn import mixture
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def bic_criterion_gmm(max_num_components, train_values, test_values):

	#Code based on: https://stackoverflow.com/questions/39920862/model-selection-for-gaussianmixture-by-using-gridsearch 
	bic = np.zeros(9)
	n = np.arange(1,max_num_components)
	models = []
	#loop through each number of Gaussians and compute the BIC, and save the model
	for i,j in enumerate(n):
	    #create mixture model with j components
	    gmm = mixture.GaussianMixture(n_components=j)
	    #fit it to the data
	    gmm.fit(train_values)
	    #compute the BIC for this model
	    bic[i] = gmm.bic(test_values)
	    #add the best-fit model with j components to the list of models
	    models.append(gmm)
	    
	plt.figure()
	plt.plot(n,bic)
	plt.ylabel("Bayesian Information Criterion (BIC)")
	plt.xlabel("Number of GMM Components")
	plt.tight_layout()
	plt.savefig("plots/bic_gmm.pdf")
	plt.show()
