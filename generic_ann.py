import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def MSE(X, Y):
	mse = np.zeros((X.shape[0], 1))
	for i in range(X.shape[0]):
		mse[i] = np.dot(X[i]-Y[i], X[i]-Y[i])
	return mse

def MSE_gra(x, y):
	return x - y
    
X = np.array([  [0,0,1],
                [0,0,0],
                [0,1,1],
                [0,1,0] ])

# X = np.array([  [0,0],
#                 [0,0],
#                 [1,1],
#                 [1,1] ])

# X.shape = (4, 3)
#Y = np.array([[0,0,1,1]]).reshape(4,1)
#Y = np.array([[0,0],[0,0],[1,1],[1,1]]).reshape(4,2)
Y = np.array([  [0,0,1],
                [0,0,0],
                [0,1,1],
                [0,1,0] ])

# Y.shape = (4, 1)

BATCH_SIZE = X.shape[0]
LEARNING_RATE = 10
N_EPOCHS = 100000

np.random.seed(1)

NEURONS = [3, 2, 3]
N_LAYERS = len(NEURONS)

weights = []
layers = ['UNDEFINED']*N_LAYERS
layers[0] = X

# Weight initialization
for i in range( N_LAYERS - 1 ):
	
	weights.append([np.random.randn(NEURONS[i], NEURONS[i+1]), np.zeros((1, NEURONS[i+1]))])
	#print i, (weights[i][0]).shape

for i in range(N_EPOCHS):
	# Forward propagation
	for l in range(0, N_LAYERS-1):
		layers[l+1] = sigmoid(np.dot(layers[l], weights[l][0]) + weights[l][1])
		#print l, layers[l].shape
	
	# Calulate loss
	loss = MSE(layers[-1], Y)
	#print 'loss.shape = ', loss.shape
	

	dL = MSE_gra(layers[-1], Y)

	# Backpropagation other layers
	for l in range(N_LAYERS-2, -1, -1):
	
		# print 'dL: ', dL.shape
		# print 'grad: ', ((layers[l+1]*(1-layers[l+1]))).shape

		dL = (layers[l+1]*(1-layers[l+1])) * dL
		
		dL_w = np.dot(dL.T, layers[l])
		
		dL_b = np.sum(dL, axis=0)
		# print 'dL_b.shape', dL_b.shape
		# print 'weights', (weights[l][1]).shape
		weights[l][0] -= LEARNING_RATE * dL_w.T
		#print (weights[l][0]).shape
		weights[l][1] -= LEARNING_RATE * dL_b

		dL = np.dot(dL, (weights[l][0]).T)
		
	if i%100 == 0 :
		print 'Loss:', np.average(loss)
		print 'Prediction:\n', layers[-1]
		print 'weights: \n', weights 
		raw_input('....')
		

