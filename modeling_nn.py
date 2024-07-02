# Neural Networks #

'''
1. perceptron
2. feed forward 
3. radial basis network 
4. deep feed forward 
5. recurrent 
6. LSTM 
7. gated recurrent unit 
8. auto encoder 
9. variational AE 
10. sparse AE 
11. denoising ae 
12. markov chain
13. hopfield network
14. boltzman machine 
15. restricted BM 
16. deep believe network 
17. deep convolutional network 
18. deep network 
19. deep conv inverse graphics network 
20. generative adverseral network 
21. liquid state machine 
22. extreme learning machine 
23. echo network machine 
24. kohoren network 
25. deep residual network 
26. support vector machine
27. neural turing machine 

Time Series Models 
Autoregression
Moving Average (MA)
Autoregressive Moving Average (ARMA)
Autoregressive Integrated Moving Average (ARIMA)
Seasonal Autoregressive Integrated Moving Average (SARIMA)
Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors (SARIMAX)
Vector Autoregression (VAR)
Vector Autoregression Moving Average (VARMA)
Vector Autoregression Moving Average with Exogenous Regressors (VARMAX)
Simple Exponential Smoothing (SES)
Holt Winter’s Exponential Smoothing (HWES)
Long Short Term Memory (LSTM)
Recurrent Neural Networks (RNN)



'''

import dependencies_1
from dependencies_1 import * 

'''
1. add regularization (l1 or l2) into models to reduce potential bias of overfitting
2. use bayesian prior distribution to initialize weights before modeling 
3. autoencoders can be used for scaling data prior to modeling 
4. amazon has python package sage maker 
5. hyper parameter optimization should be done; methods include exhaustive grid, random parameter opt in sckit 

'''


def knn_model(X,y,x_test,num_neigh,num_leaves,model_type): # works for classification targets  
	knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = num_neigh,leaf_size = num_leaves,p = model_type)
	# model type : 1 (manhattan_distance) and 2(euclidean_distance)
	knn.fit(X, y)
	pred = knn.predict_proba(x_test)
	return pred


def naive_bayes_model(X,y,x_test): # works for classification targets 
	gnb = sklearn.naive_bayes.GaussianNB(priors = [0.25,0.25,0.5])
	gnb.fit(X,y)
	pred = gnb.predict(x_test)
	return pred


def svc_model(X,y,x_test,reg_parm,kernel_parm):
	svc = sklearn.svm.SVC(C = reg_parm, kernel = kernel_parm,random_state= 910)
	svc.fit(X,y)
	pred = svc.predict(dataset)
	return pred 


def svr_model(X,Y,X_test):
	X = numpy.array(X)

	Y = numpy.array(Y)

	X_test = numpy.array(X_test)

	svr = sklearn.svm.SVR()
	svr.fit(X,Y)
	predict = svr.predict(X_test)
	get_parms = svr.get_params
	scores = svr.score(X,Y)

	return predict, get_parms, scores


def mlp_neural_net_reg(X_train,y_train,X_test,y_test,layers,l1,l2,l3,epoch_num):

	model = Sequential()

	if layers == 1:
		model.add(Dense(X_train.shape[1],input_dim = X_train.shape[1]))
		model.add(Dense(5,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(1,activation = 'relu'))
		model.compile(optimizer = 'RMSprop',loss = 'mse', metrics = ['mae'])

	elif layers == 2:
		model.add(Dense(X_train.shape[1],input_dim = X_train.shape[1]))
		model.add(Dense(l1,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(l2,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(1,activation = act3))
		model.compile(optimizer = 'RMSprop',loss = 'mse', metrics = ['mae'])

	else:
		model.add(Dense(X_train.shape[1],input_dim = X_train.shape[1]))
		model.add(Dense(l1,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(l2,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(l3,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))	
		model.add(Dense(1,activation = 'relu'))	
		model.compile(optimizer = 'RMSprop',loss = 'mse', metrics = ['mae'])


	batch = len(X_train) * 0.1
	batch = round(batch)

	model.fit(X_train, y_train, batch_size= batch, epochs= 5, verbose=0, validation_data=(X_test, y_test))
	evaluation = model.evaluate(X_test,y_test,batch_size = batch)
	predictions = model.predict(X_test, batch_size= batch)
	predictions_class = model.predict_classes(X_test,batch_size=batch)

	return predictions, evaluation



def mlp_neural_net_class(X_train,y_train,X_test,y_test,layers,l1,l2,l3,epoch_num):

	model = Sequential()

	if layers == 1:
		model.add(Dense(X_train.shape[1],input_dim = X_train.shape[1]))
		model.add(Dense(5,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(1,activation = 'relu'))
		model.compile(optimizer = 'RMSprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])

	elif layers == 2:
		model.add(Dense(X_train.shape[1],input_dim = X_train.shape[1]))
		model.add(Dense(l1,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(l2,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(1,activation = act3))
		model.compile(optimizer = 'RMSprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])

	else:
		model.add(Dense(X_train.shape[1],input_dim = X_train.shape[1]))
		model.add(Dense(l1,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(l2,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(l3,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))	
		model.add(Dense(1,activation = 'relu'))	
		model.compile(optimizer = 'RMSprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])


	batch = len(X_train) * 0.1
	batch = round(batch)

	model.fit(X_train, y_train, batch_size= batch, epochs= 5, verbose=0, validation_data=(X_test, y_test))
	evaluation = model.evaluate(X_test,y_test,batch_size = batch)
	predictions = model.predict(X_test, batch_size= batch)
	predictions_class = model.predict_classes(X_test,batch_size=batch)

	return predictions, predictions_class, evaluation


def mlp_neural_net_bin_reg(X_train,y_train,X_test,y_test,layers,l1,l2,l3,epoch_num):

	model = Sequential()

	if layers == 1:
		model.add(Dense(X_train.shape[1],input_dim = X_train.shape[1]))
		model.add(Dense(5,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(1,activation = 'relu'))
		model.compile(optimizer = 'RMSprop',loss = 'binary_crossentropy', metrics = ['accuracy'])

	elif layers == 2:
		model.add(Dense(X_train.shape[1],input_dim = X_train.shape[1]))
		model.add(Dense(l1,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(l2,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(1,activation = act3))
		model.compile(optimizer = 'RMSprop',loss = 'binary_crossentropy', metrics = ['accuracy'])

	else:
		model.add(Dense(X_train.shape[1],input_dim = X_train.shape[1]))
		model.add(Dense(l1,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(l2,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))
		model.add(Dense(l3,kernel_initializer = 'uniform', activation = 'relu'))
		model.add(Dropout(0.05))	
		model.add(Dense(1,activation = 'relu'))	
		model.compile(optimizer = 'RMSprop',loss = 'binary_crossentropy', metrics = ['accuracy'])


	batch = len(X_train) * 0.1
	batch = round(batch)

	model.fit(X_train, y_train, batch_size= batch, epochs= 5, verbose=0, validation_data=(X_test, y_test))
	evaluation = model.evaluate(X_test,y_test,batch_size = batch)
	predictions = model.predict(X_test, batch_size= batch)
	predictions_class = model.predict_classes(X_test,batch_size=batch)

	return predictions, predictions_class, evaluation


def cnn_neural_net(x_train,layers,
					l1,k1,act1,
					l2,k2,act2,d1,
					l3,k3,act3,
					l4,k4,act4,d2,
					d3,l5,optim,loss_f):

	model = Sequential()

	if layers == 1:

		model.add(Conv2D(l1, k1 , padding='same', input_shape=(x_train.shape[0],x_train.shape[1],1), activation = act1))
		model.add(Conv2D(l2, k2 , padding='same', activation = act2))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(d1))

		model.add(Flatten())
		model.add(Dense(l3,activation = 'softmax'))
		model.add(Dropout(d2))

		model.compile(optimizer = optim, loss = loss_f, metrics = ['accuracy'])

		return model

	else:

		model.add(Conv2D(l1, k1 , padding='same', input_shape=(x_train.shape[0],x_train.shape[1],1), activation = act1))
		model.add(Conv2D(l2, k2 , padding='same', activation = act2))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(d1))

		model.add(Conv2D(l3, k3 , padding='same', input_shape=(x_train.shape[0],x_train.shape[1],1), activation = act3))
		model.add(Conv2D(l4, k4 , padding='same', activation = act4))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(d2))

		model.add(Flatten())
		model.add(Dense(l5,activation = 'softmax'))
		model.add(Dropout(d3))

		model.compile(optimizer = optim, loss = loss_f, metrics = ['accuracy'])
		
		return model


def lstm_net(x_train,layers,l1,num_lags,act1,d1,l2,act2,d2,l3,act3,d3,l4,act4,l5,optim,loss_f):

	model = Sequential()

	if layers == 1:
		model.add(LSTM(l1, batch_input_shape= (x_train.shape[0], num_lags, x_train.shape[1]), stateful=True, kernel_initializer = 'random_uniform')) 
		model.add(Dense(l2, activation= act1))
		model.add(Dropout(d1))
		model.add(Dense(l3,activation = act2))
		model.compile(loss=loss_f, optimizer=optim, metrics = ['accuracy'])

		return model 

	elif layers == 2:

		model.add(LSTM(l1, batch_input_shape= (x_train.shape[0], num_lags, x_train.shape[1]), stateful=True, kernel_initializer = 'random_uniform')) 
		model.add(Dense(l2, activation= act1))
		model.add(Dropout(d1))

		model.add(Dense(l3, activation= act2))
		model.add(Dropout(d2))

		model.add(Dense(l4,activation = act3))
		model.compile(optimizer = optim,loss = loss_f, metrics = ['accuracy'])

		return model 

	else:
		model.add(LSTM(l1, batch_input_shape= (x_train.shape[0], num_lags, x_train.shape[1]), stateful=True, kernel_initializer = 'random_uniform')) 
		model.add(Dense(l2, activation= act1))
		model.add(Dropout(d1))

		model.add(Dense(l3, activation= act2))
		model.add(Dropout(d2))


		model.add(Dense(l4, activation= act3))
		model.add(Dropout(d3))

		model.add(Dense(l5,activation = act4))
		model.compile(optimizer = optim,loss = loss_f, metrics = ['accuracy'])

		return model 


def model_construct(data,neurons1,RELU_factor1,neurons2,RELU_factor2,neurons3,RELU_factor3,neurons4,
					neurons5,RELU_factor4,drop1,neurons6,RELU_factor5,drop2,neurons7,RELU_factor6,
					drop3,neurons8,optim_g,loss_g,loss_d,optim_d):

	np.random.seed(10)

	generator = Sequential()
	generator.add(Dense(neurons1, input_dim=data.shape[1], kernel_initializer=initializers.RandomNormal(stddev=0.05)))
	generator.add(LeakyReLU(RELU_factor1))

	generator.add(Dense(neurons2))
	generator.add(LeakyReLU(RELU_factor2))

	generator.add(Dense(neurons3))
	generator.add(LeakyReLU(RELU_factor3))

	generator.add(Dense(neurons4, activation='tanh'))
	generator.compile(loss=loss_g, optimizer=optim_g)

	
	discriminator = Sequential()
	discriminator.add(Dense(neurons5, input_dim=data.shape[1], kernel_initializer=initializers.RandomNormal(stddev=0.02)))
	discriminator.add(LeakyReLU(RELU_factor4))
	discriminator.add(Dropout(drop1))

	discriminator.add(Dense(neurons6))
	discriminator.add(LeakyReLU(RELU_factor5))
	discriminator.add(Dropout(drop2))

	discriminator.add(Dense(neurons7))
	discriminator.add(LeakyReLU(RELU_factor6))
	discriminator.add(Dropout(drop3))

	discriminator.add(Dense(neurons8, activation='sigmoid'))
	discriminator.compile(loss=loss_d, optimizer=optim_d)

	return generator, discriminator


def get_gan_network(discriminator, data, generator, optimizer):

	discriminator.trainable = False
	gan_input = Input(shape=(data.shape[1],))
	x = generator(gan_input)
	gan_output = discriminator(x)
	    
	gan = Model(inputs=gan_input, outputs=gan_output)
	gan.compile(loss='binary_crossentropy', optimizer=optimizer)

	return gan


def train_gan(epochs, batch_size):

	batch_count = x_train.shape[0] / batch_size

	#2
	adam = get_optimizer()
	generator = model_construct.generator(adam)
	discriminator = model_construct.discriminator(adam)
	gan = get_gan_network(discriminator, random_dim, generator, adam)

	#3
	for e in range(1, epochs+1):
		print('-'*15, 'Epoch %d' % e, '-'*15)
		for i in tqdm(range(int(batch_count))):
	            
			#4
			noise = np.random.normal(0, 1, size=[batch_size, random_dim])
			image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

			#5
			generated_images = generator.predict(noise)
			X = np.concatenate([image_batch, generated_images])

			#6
			y_dis = np.zeros(2*batch_size)
			y_dis[:batch_size] = 0.9

			#7
			discriminator.trainable = True
			discriminator.train_on_batch(X, y_dis)

			#8
			noise = np.random.normal(0, 1, size=[batch_size, random_dim])
			y_gen = np.ones(batch_size)
			discriminator.trainable = False
			gan.train_on_batch(noise, y_gen)
		        
		if e == 1 or e % 20 == 0:
			plot_generated_images(e, generator)



def model_report(model):

	shape = model.output_shape
	shape = str(shape)
	summary = model.summary()
	config = model.get_config()
	weights = model.get_weights()

	return shape, summary, config, weights 



def model_training(x_train,y_train,x_test,y_test,coef,epoch_num):

	batch = len(x_train) * coef 

	model.fit(x_train, y_train, batch_size= batch, epochs= epoch_num, verbose=0, validation_data=(x_test, y_test))

	evaluation = model.evaluate(x_test,y_test,batch_size = batch)

	return evaluation


def learning_rate(learn_rate, momentum_rate, decay_rate ):

	sgd = SGD(lr=learn_rate, momentum=momentum_rate, decay=decay_rate, nesterov=False)
	return sgd 


def optimization(learn_rate,b1,b2,rho_coef):
		sgdopt = keras.optimizers.SGD(learning_rate=learn_rate, momentum=0.0, nesterov=False)
		rmspropopt = keras.optimizers.RMSprop(learning_rate=learn_rate, rho=rho_coef)
		adagradopt = keras.optimizers.Adagrad(learning_rate=learn_rate)
		adadeltaopt = keras.optimizers.Adadelta(learning_rate=learn_rate, rho=rho_coef)
		adamopt = keras.optimizers.Adam(learning_rate=learn_rate, beta_1=b1, beta_2=b2, amsgrad=False)
		adamaxopt = keras.optimizers.Adamax(learning_rate=learn_rate, beta_1=b1, beta_2=b2)
		nadamoopt = keras.optimizers.Nadam(learning_rate=learn_rate, beta_1=b1, beta_2=b2)

		return sgdopt, rmspropopt, adagradopt, adadeltaopt, adamopt, adamaxopt, nadamoopt


def model_predict(dataset,coef):

	batch = len(dataset) * coef 

	predictions = model.predict(dataset, batch_size= batch)
	predictions_class = model.predict_classes(dataset,batch_size=batch)

	return predictions, predictions_class


def model_save(model_name):
	model.save('%s.h5' %(model_name))


def model_load(model_name):

	my_model = load_model('%s.h5' %(model_name))
	return my_model 


def markov_chain_model(a,number_states):
	model = sklearn.mixture.GaussianMixture(n_components=number_states, # 3 for the above example # 
											covariance_type="full", 
											n_init=100, 
											random_state=910)
	model.fit(a)

	means = model.means_
	covar = model.covariances_
	return model,means,covar


def markov_chain_predict(model,X_test):
	labels = model.predict(X_test)
	return labels 


def markov_chain_plot(X,labels):
	X['labels']= labels 
	X0 = X[X['labels']== 0] 
	X1 = X[X['labels']== 1] 
	X2 = X[X['labels']== 2] 

	plt.scatter(X0[0], X0[1], c ='r') 
	plt.scatter(X1[0], X1[1], c ='yellow') 
	plt.scatter(X2[0], X2[1], c ='g') 


def opt_returns(dataset):

		mr = pandas.DataFrame()

	# compute monthly returns
		for s in dataset.columns:
			date = dataset.index[0]
			pr0 = dataset[s][date] 
			for t in range(1,len(dataset.index)):
				date = dataset.index[t]
				pr1 = dataset[s][date]
				ret = (pr1-pr0)/pr0
				mr.set_value(date,s,ret)
				pr0 = pr1

		# symbols #
		symbols = mr.columns

		# df to array #
		return_data = mr.as_matrix().T

		r = np.asarray(np.mean(return_data, axis=1)) #return
		C = np.asmatrix(np.cov(return_data)) #covariance

		return r,C


def min_opt_model(dataset,r,C,min_req_return): 

# build in stop loss limits as constraints # 
# build in VAR score into risk assessment # 
    	
    # Number of variables
	n = dataset.shape[1]

	# The variables vector
	x = Variable(n)

	# The return
	ret = r.T*x

	# The risk in xT.Q.x format
	risk = quad_form(x, C)

	# The core problem definition with the Problem class from CVXPY
	prob = Problem(Minimize(risk), [sum(x)==1, ret >= min_req_return, x >= 0]) # constraints in []

	prob.solve()
	print("status:", prob.status)
	print("Expected Return:", ret.value)
	print("Expected Risk:", risk.value)

	status_opt = prob.status
	weights = x.value
	returns = ret.value
	risks = risk.value

	return status_opt, weights, returns, risks

def max_opt_model(dataset,r,C,max_risk): # max_risk = 0 is a risk parity portfolio

# build in stop loss limits as constraints # 
# build in VAR score into risk assessment # 
    	
    # Number of variables
	n = dataset.shape[1]

	# The variables vector
	x = Variable(n)

	# The return
	ret = r.T*x

	# The risk in xT.Q.x format
	risk = quad_form(x, C)

	# The core problem definition with the Problem class from CVXPY
	prob = Problem(Maximize(ret), [sum(x)==1, risk <= max_risk, x >= 0]) # constraints in []

	prob.solve()
	print("status:", prob.status)
	print("Expected Return:", ret.value)
	print("Expected Risk:", risk.value)

	status_opt = prob.status
	weights = x.value
	returns = ret.value
	risks = risk.value

	return status_opt, weights, returns, risks



def optimal_portfolio(returns,n_iter):  # future iter contains all random portfolios 
# https://www.quantopian.com/posts/the-efficient-frontier-markowitz-portfolio-optimization-in-python # 
	n = len(returns)  
	returns = np.asmatrix(returns)  
	N = n_iter  
	mus = [10**(5.0 * t/N - 1.0) for t in range(N)]  
	# Convert to cvxopt matrices  
	S = opt.matrix(np.cov(returns))  
	pbar = opt.matrix(np.mean(returns, axis=1))  
	# Create constraint matrices  
	G = -opt.matrix(np.eye(n))   # negative n x n identity matrix  
	h = opt.matrix(0.0, (n ,1))  
	A = opt.matrix(1.0, (1, n))  
	b = opt.matrix(1.0)  
	# Calculate efficient frontier weights using quadratic programming  
	portfolios = [solvers.qp(a*S, -pbar, G, h, A, b)['x']  
						for a in mus]  
	## CALCULATE RISKS AND RETURNS FOR FRONTIER  
	returns = [blas.dot(pbar, x) for x in portfolios]  
	risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]  
	## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE  
	m1 = np.polyfit(returns, risks, 2)  
	x1 = np.sqrt(m1[2] / m1[0])  
	# CALCULATE THE OPTIMAL PORTFOLIO  
	wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  
	weights = numpy.asarray(wt)

	return weights, returns, risks 


def plot_efficient_front(risks,returns):
	plt.ylabel('mean')  
	plt.xlabel('std')  
	plt.plot(risks, returns, 'y-o')  



def initialize(context):  
	'''  
	Called once at the very beginning of a backtest (and live trading).  
	Use this method to set up any bookkeeping variables.  
	The context object is passed to all the other methods in your algorithm.

	Parameters

	context: An initialized and empty Python dictionary that has been  
	augmented so that properties can be accessed using dot  
	notation as well as the traditional bracket notation.  
	Returns None  
	''' 

	    # Turn off the slippage model  
	set_slippage(slippage.FixedSlippage(spread=0.0))  
	    # Set the commission model (Interactive Brokers Commission)  
	set_commission(commission.PerShare(cost=0.01, min_trade_cost=1.0))  
	context.tick = 0  


def handle_data(context, data):  
	'''  
	Called when a market event occurs for any of the algorithm's securities. 
	Parameters
	data: A dictionary keyed by security id containing the current state of the securities in the algo's universe.
	context: The same context object from the initialize function. Stores the up to date portfolio as well as any state variables defined.
	Returns None  
	'''  
	# Allow history to accumulate 100 days of prices before trading  
	# and rebalance every day thereafter.  
	context.tick += 1  
	if context.tick < 100:  
		return  
	# Get rolling window of past prices and compute returns  
	prices = history(100, '1d', 'price').dropna()  
	returns = prices.pct_change().dropna()  
	try:  
		# Perform Markowitz-style portfolio optimization  
		weights, _, _ = optimal_portfolio(returns.T)  
		# Rebalance portfolio accordingly  
		for stock, weight in zip(prices.columns, weights):  
			order_target_percent(stock, weight)  
	except ValueError as e:  
		# Sometimes this error is thrown  
		# ValueError: Rank(A) < p or Rank([P; A; G]) < n  
		pass  



def cluster_model_kmeans(X,n_clust,graph_title,x_label,y_label):

    kmeans = KMeans(n_clusters = n_clust, init = 'k-means++', random_state = 910)
    y_kmeans = kmeans.fit_predict(X)
    labels = kmeans.labels_
    X = numpy.array(X)

    # Visualising the clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 10, c = 'magenta', label = 'Cluster 5')
    plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 10, c = 'yellow', label = 'Cluster 6')
    plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 10, c = 'purple', label = 'Cluster 7')
    plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s = 10, c = 'grey', label = 'Cluster 8')
    plt.scatter(X[y_kmeans == 8, 0], X[y_kmeans == 8, 1], s = 10, c = 'black', label = 'Cluster 9')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
    plt.title('%s' %(graph_title))
    plt.xlabel('%s' %(x_label))
    plt.ylabel('%s' %(y_label))
    plt.legend()
    plt.show()

    clust_1 = X[y_kmeans == 0, 0]
    clust_2 = X[y_kmeans == 1, 0]
    clust_3 = X[y_kmeans == 2, 0]
    clust_4 = X[y_kmeans == 3, 0]
    clust_5 = X[y_kmeans == 4, 0] 
    clust_6 = X[y_kmeans == 5, 0]
    clust_7 = X[y_kmeans == 6, 0]
    clust_8 = X[y_kmeans == 7, 0]
    clust_9 = X[y_kmeans == 8, 0] 

    if n_clust == 1:
        return clust_1 

    elif n_clust == 2:
        return clust_1, clust_2 

    elif n_clust == 3:
        return clust_1, clust_2, clust_3 

    elif n_clust == 4:
        return clust_1, clust_2, clust_3, clust_4 

    elif n_clust == 5:
         return clust_1, clust_2, clust_3, clust_4, clust_5 
    
    elif n_clust == 6:
         return clust_1, clust_2, clust_3, clust_4, clust_5, clust_6 
         
    elif n_clust == 7:
         return clust_1, clust_2, clust_3, clust_4, clust_5, clust_6, clust_7 

    elif n_clust == 8:
         return clust_1, clust_2, clust_3, clust_4, clust_5, clust_6, clust_7, clust_8 

    else: 
        return clust_1, clust_2, clust_3, clust_4, clust_5, clust_6, clust_7, clust_8, clust_9 



def linear_reg(X,Y,x_test):

	X = numpy.array(X)

	Y = numpy.array(Y)

	x_test = numpy.array(x_test)

	reg = sklearn.linear_model.LinearRegression().fit(X,Y)
	scores = reg.score(X,Y)
	coef = reg.coef_
	inter = reg.intercept_
	predict = reg.predict(x_test)

	reg_residuals = analysis_7.autocorr_test(reg.resid)
	heterosked = analysis_7.heteroskedasticity_test(reg.resid, reg.model.exog)

	return scores, coef, inter, predict, reg_residuals, heterosked


def logistic_reg(X,Y,X_test,penalty_value,threshold):

	X = numpy.array(X)

	Y_bin = Y.apply(lambda x: 1 if x > threshold  else 0) 
	Y_bin = numpy.array(Y_bin)

	X_test = numpy.array(X_test)

	reg = sklearn.linear_model.LogisticRegression(penalty = penalty_value, random_state = 910 )
	reg.fit(X,Y_bin)
	predict = reg.predict(X_test)
	predict_prob = reg.predict_proba(X_test)
	predict_prob = predict_prob[:,1]
	predict_prob = predict_prob.reshape(-1,1)

	scores = reg.score(X,Y_bin)

	reg_residuals = analysis_7.autocorr_test(reg.resid)
	heterosked = analysis_7.heteroskedasticity_test(reg.resid, reg.model.exog)


	return predict, predict_prob, scores, Y_bin, reg_residuals, heterosked
	

def decision_tree_classifier(X,Y,X_test,depth_level):
	X = numpy.array(X)

	Y = numpy.array(Y)

	X_test = numpy.array(X_test)

	clf = tree.DecisionTreeClassifier(random_state = 910, max_depth = depth_level)
	clf = clf.fit(X, Y)
	predict = clf.predict(X_test)
	predict_prob = clf.predict_proba(X_test)
	
#	plot = tree.plot_tree(clf.fit(X, Y)) 
	plot = tree.export_graphviz(clf)

	return predict, predict_prob, plot



def decision_tree_reg(X,Y,X_test,depth_level):
	X = numpy.array(X)

	Y = numpy.array(Y)

	X_test = numpy.array(X_test)

	clf = tree.DecisionTreeRegressor(random_state = 910, max_depth = depth_level)
	clf = clf.fit(X, Y)
	predict = clf.predict(X_test)

	return predict


def grad_boost_tree(X,Y,X_test,estimators,subsample_n,split_n):

	X = numpy.array(X)

	Y = numpy.array(Y)

	X_test = numpy.array(X_test)
	
	clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators = estimators, subsample = subsample_n, min_samples_split = split_n)
	clf.fit(X,Y)
	params = clf.get_params
	predict = clf.predict(X_test)
	predict_prob = clf.predict_proba(X_test)
	scores = clf.score(X,Y)

	return params, predict, predict_prob, scores


def random_forest(X,Y,X_test,estimators, depth_level, split_n, leaf_n):

	X = numpy.array(X)

	Y = numpy.array(Y)

	X_test = numpy.array(X_test)

	clf = sklearn.ensemble.RandomForestClassifier(n_estimators = estimators, max_depth = depth_level, min_samples_split = split_n, 
												min_samples_leaf = leaf_n, random_state = 910 )

	clf.fit(X, Y)
	feature_impor = clf.feature_importances_
	predict = clf.predict(X_test)
	predict_logs = clf.predict_log_proba(X_test)
	predict_proba = clf.predict_proba(X_test)
	scores = clf.score(X,Y)

	return feature_impor, predict, predict_logs, predict_proba, scores 


# def semi_supervised_models():
# https://scikit-learn.org/stable/modules/label_propagation.html#label-propagation #


def ensemble_modeling(model1,model2,model3,x_test,y_test):
	classifier = VotingClassifier(estimators=[('kn', model1), ('rf', model2), ('dt', model3)], voting='hard')  # ensemble model

	for model in ([model1, model2, model3, classifier]):
		scores = cross_val_score(model, x_test, y_test, cv=3,  scoring='accuracy')
		print("Accuracy: " % scores.mean()) # mean of the ensemble model


