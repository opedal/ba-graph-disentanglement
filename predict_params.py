from generate_data import generate_ba_nx_graphs, generate_er_nx_graphs
import numpy as np
from graph_scoring import GraphScorer
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, recall_score, precision_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr
import pandas as pd

def get_rf_grid(n_ests=[50,100, 200, 500],
                max_features=['auto', 'sqrt', 'log2'],
                max_depth=[4, 5, 6, 7, 8],
                cv_fold_num=5,
                ):
    param_grid = {
        'n_estimators': n_ests,
        'max_features': max_features,
        'max_depth': max_depth,
    }
    return  GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=cv_fold_num)

def get_mlp(feature_num=13,output_dim=3,opt='adam',loss='mean_squared_error'):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    mlp = Sequential()
    mlp.add(Dense(32, input_shape=(feature_num,)))
    mlp.add(Dense(32, activation='tanh'))
    mlp.add(Dense(64, activation='tanh'))
    mlp.add(Dense(output_dim))

    mlp.compile(optimizer=opt, loss=loss,metrics=['accuracy'])
    return mlp

def get_mlp2(feature_num=13,output_dim=3,opt='adam',loss='mean_squared_error'):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    mlp = Sequential()
    mlp.add(Dense(80, input_shape=(feature_num,)))
    mlp.add(Dense(40, activation='tanh'))
    mlp.add(Dense(output_dim))

    mlp.compile(optimizer=opt, loss=loss,metrics=['accuracy'])
    return mlp

class ParamPredictor:

    def __init__(self, model=None,model_name=''):
        if model is None:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
        self._model = model
        self.model_name = model_name

    def set_model(self,new_model,new_model_name):
        """
        :new_model: New ML model, should have functions predict() and fit()
        """
        self._model = new_model
        self.model_name = new_model_name

    def fit_model(self,X,y,**args):
        self._model.fit(X,y,**args)

    def predict_params(self, Xtst):
        return self._model.predict(Xtst)

def get_feature_vec(graph,scorer):
    features = []
    for score_name in scorer.available_scores():
        features.append(scorer.score(graph, score_name))
    return features

def preprocess(graphs,scorer):
    feature_vecs = np.array([get_feature_vec(g,scorer) for g in graphs])
    return feature_vecs

def get_fit_predictor(X,y,model=None,model_name='',**args):
    predictor = ParamPredictor(model=model, model_name=model_name)

    if 'model_fit_args' in args:
        predictor.fit_model(X, y, **args['model_fit_args'])
    else:
        predictor.fit_model(X, y)
    return predictor

def analyze_alpha_preds(X,y,Xtst,ytst,scorer,model,model_name,alpha_idx=2,**args):

    predictor = ParamPredictor(model=model, model_name=model_name)

    if 'model_fit_args' in args:
        predictor.fit_model(X, y, **args['model_fit_args'])
    else:
        predictor.fit_model(X, y)

    yprd = predictor.predict_params(Xtst)

    alpha_pred = yprd[:,alpha_idx]
    alpha_true = ytst[:,alpha_idx]

    print("current model is ",predictor.model_name)
    print("total mse is ", mean_squared_error(yprd, ytst))
    print("mse for alpha is ", mean_squared_error(alpha_pred, alpha_true))
    print("mean absolute error for alpha is ", mean_absolute_error(alpha_pred, alpha_true))
    r, _ = pearsonr(alpha_true, alpha_pred)
    print("pearson r correlation between true and predicted alpha is ",r)
    results = pd.DataFrame({"score": [mean_squared_error(yprd, ytst),
                                      mean_squared_error(alpha_pred, alpha_true),
                                      r
                                      ],
                            "param": ['total_mse', 'alpha_mse','pearsonr'],
                            "model_name": [model_name, model_name,model_name]})

    results.to_csv("results" + model_name + ".csv")
    alphas = [alpha_pred, alpha_true]
    plt.clf()
    plt.hist(alphas, 22, histtype='bar',color=['deepskyblue','xkcd:saffron'], label=['pred','true'])
    plt.title("alpha histograms for "+predictor.model_name)
    plt.savefig("alpha_hists_"+predictor.model_name + ".png")

    plt.clf()
    plt.scatter(x=ytst[:, alpha_idx], y=yprd[:, alpha_idx],color="deepskyblue")
    plt.ylabel("predicted alpha")
    plt.xlabel("true alpha")
    plt.title("true versus predicted alpha for "+predictor.model_name)
    plt.savefig("predicted_alpha_"+predictor.model_name + ".png")

    er_alpha_preds = predict_on_er(predictor,scorer)
    yprd = predictor.predict_params(preprocess(Xraw_tst,scorer))
    ba_alpha_preds = yprd[:, 2]

    plt.clf()
    plt.hist([er_alpha_preds,ba_alpha_preds],density=True,color=['deepskyblue','xkcd:saffron'],label=['ER','BA'])
    plt.title("alpha predictions for ER graphs")
    plt.legend()
    plt.savefig("predicted_alpha_for_er_"+predictor.model_name + ".png")

def predict_on_er(predictor,scorer,num_graphs=600,n=10,alpha_idx=2):
    er_graphs = generate_er_nx_graphs(num_graphs,n)
    er_graph_features = preprocess(er_graphs,scorer)
    preds = predictor.predict_params(er_graph_features)
    alpha = preds[:,alpha_idx]
    return alpha

if __name__ == '__main__':

    scorer = GraphScorer()
    num_ba_params = 3
    N = 10

    Xraw, y = generate_ba_nx_graphs(num_graphs=400,n=N,num_ba_params=num_ba_params)
    Xraw_tst, ytst = generate_ba_nx_graphs(num_graphs=100,n=N,num_ba_params=num_ba_params)

    X = preprocess(Xraw, scorer)
    Xtst = preprocess(Xraw_tst, scorer)

    rf_model = get_rf_grid()
    analyze_alpha_preds(X,y,Xtst,ytst,scorer,model=rf_model,model_name='rfgrid')

    model = get_mlp2(feature_num=scorer.num_scores(), output_dim=num_ba_params)

    analyze_alpha_preds(X,y,Xtst,ytst,scorer,model=model,
                          model_name='two layer mlp 2',
                          model_fit_args={'epochs':400,
                                          'batch_size':32,
                                          'validation_split':0.2})
