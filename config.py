ENTRY_NUMBER_LAST_DIGIT = 2 # change with yours
ENTRY_NUMBER = '2022MT61202'
PRE_PROCESSING_CONFIG ={
    "hard_margin_linear" : {
        "use_pca" : None,
    },

    "hard_margin_rbf" : {
        "use_pca" : None,
    },

    "soft_margin_linear" : {
        "use_pca" : None,
    },

    "soft_margin_rbf" : {
        "use_pca" : None,
    },

    "AdaBoost" : {
        "use_pca" : None,
    },

    "RandomForest" : {
        "use_pca": None,
    }
}

SVM_CONFIG = {
    "hard_margin_linear" : {
        "C" : 1e9,
        "kernel" : 'linear',
        "val_score" : 0.70, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
    },
    "hard_margin_rbf" : {
        "C" : 1e9,
        "kernel" : 'rbf',
        "val_score" : 0.891919565678854, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
        
        ## Gamma == 0.1
    },

    "soft_margin_linear" : {
        "C" : 0.001, # add your best hyperparameter
        "kernel" : 'linear',
        "val_score" : 0.8277923540339666, # add the validation score you get on val set for the set hyperparams.
                         # Diff in your and our calculated score will results in severe penalites
        # add implementation specific hyperparams below (with one line explanation)
        
    },

    "soft_margin_rbf" : {
         "C" : 10, # add your best hyperparameter
         "kernel" : 'rbf',
         "val_score" : 0.891919565678854, # add the validation score you get on val set for the set hyperparams.
                          # Diff in your and our calculated score will results in severe penalites
         # add implementation specific hyperparams below (with one line explanation)
         
         ## Gamma == 0.1
     }
}

ENSEMBLING_CONFIG = {
    'AdaBoost':{
        'num_trees' : 400,
        "val_score" : 0.8375592182918292,
    },

    'RandomForest':{
        'num_trees' : 20,
        "val_score" : 0.7475151939660051,
    }
}
