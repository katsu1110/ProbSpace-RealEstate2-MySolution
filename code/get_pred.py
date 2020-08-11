import numpy as np

def get_oof_ypred(model, x_val, x_test, modelname="lgb", task="regression"):  
    """
    get oof and target predictions
    """
    sklearns = ["xgb", "catb", "linear", "knn"]
    if task == "multiclass":
        sklearns.append("lgb")

    if task == "binary": # classification
        # sklearn API
        if modelname in sklearns:
            oof_pred = model.predict_proba(x_val)
            y_pred = model.predict_proba(x_test)
            oof_pred = oof_pred[:, 1]
            y_pred = y_pred[:, 1]
        else:
            oof_pred = model.predict(x_val)
            y_pred = model.predict(x_test)

            # NN specific
            if modelname == "nn":
                oof_pred = oof_pred.ravel()
                y_pred = y_pred.ravel()        

    elif task == "multiclass":
        # sklearn API
        if modelname in sklearns:
            oof_pred = model.predict_proba(x_val)
            y_pred = model.predict_proba(x_test)
        else:
            oof_pred = model.predict(x_val)
            y_pred = model.predict(x_test)

        # oof_pred = np.argmax(oof_pred, axis=1)
        # y_pred = np.argmax(y_pred, axis=1)

    elif task == "regression": # regression
        oof_pred = model.predict(x_val)
        y_pred = model.predict(x_test)

        # NN specific
        if modelname == "nn":
            oof_pred = oof_pred.ravel()
            y_pred = y_pred.ravel()

    return oof_pred, y_pred