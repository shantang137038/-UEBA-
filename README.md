# 基于UEBA的用户上网异常行为分析
数据来源于https://www.datafountain.cn/competitions/520
## 特征工程

    from sklearn.model_selection import KFold
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, make_scorer, roc_curve, f1_score, auc, log_loss, r2_score,mean_squared_error
    import pandas as  pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import matplotlib.pyplot as plt
编码特征包括["account","group","IP","url","vlan"]

时间提取小时数据

其余特征处理包括：提取ip的每一段数据，account的长度、group的长度

  特征信息提取：
  
    data["hour"] = pd.to_datetime(data["time"]).dt.hour
    del data["time"]
    def extract_ip_0(x):
        return int(x.split(".")[0])
    def extract_ip_1(x):
        return int(x.split(".")[1])
    def extract_ip_2(x):
        return int(x.split(".")[2])
    def extract_ip_3(x):
        return int(x.split(".")[3])
    data["IP_0"] = data["IP"].apply(extract_ip_0)
    data["IP_1"] = data["IP"].apply(extract_ip_1)
    data["IP_2"] = data["IP"].apply(extract_ip_2)
    data["IP_3"] = data["IP"].apply(extract_ip_3)
    def len_account(x):
        return len(x.split("@")[0])
    data["account_len"] = data["account"].apply(len_account)
    data["group_len"] = data["group"].apply(len_account)
    
  编码：

    cat_feature= ["account","group","IP","url","vlan"]
    for cat_f in cat_feature:
        lab_ecd = LabelEncoder()
        data[cat_f+"_ecd"] = lab_ecd.fit_transform(data[cat_f])
   
  删除不要的列：
  
    use_data = data.copy()
    use_data.drop(cat_feature,axis=1,inplace=True)
    use_data.drop(["port","switchIP","ret","id"],axis=1,inplace=True)
    use_data["ret"] = data["ret"]
    use_data.info()

## 模型
选择的是xgboost，参数默认，未寻优。

    k_fold = 2
    seed = 1000
    kf = KFold(n_splits=k_fold,shuffle=True,random_state=seed)
    for ii,[train_index,test_index] in enumerate(kf.split(use_data)):
        train_data = use_data.iloc[train_index].reset_index(drop=True)
        test_data = use_data.loc[test_index].reset_index(drop=True)
        trainx,trainy,test_x,test_y = train_data.iloc[:,:-1],train_data.iloc[:,-1],test_data.iloc[:,:-1],test_data.iloc[:,-1]
        train_x,val_x,train_y,val_y = train_test_split(trainx, trainy, test_size=0.4)
        print("训练数据长度{}，训练数据标签长度{}，验证数据长度{}，验证数据标签长度{}".format(train_x.shape[0],train_y.shape[0],val_x.shape[0],val_y.shape[0]))
        model = XGBRegressor()
        model.fit(train_x,train_y,early_stopping_rounds=1,eval_set=[(val_x, val_y)], verbose=5)
        mape = np.sqrt(np.sum((test_y - model.predict(test_x))**2)/len(test_y))
        print(mape)
        print('r2:',r2_score(test_y ,model.predict(test_x)))
        print("mean_squared_error",mean_squared_error(test_y ,model.predict(test_x)))
        plt.plot(np.array(test_y)[::200],color="b")
        plt.plot(model.predict(test_x)[::200],color="r")
        plt.show()
        
## 效果
交叉验证后，R2在0.79左右，rmse在0.0098左右

![image](https://user-images.githubusercontent.com/57713334/205566172-4873f1dc-5170-4d48-bb76-b304eb93b624.png)

一些数据高点和低点预测的不是很准，数据可以考虑做一个数据预处理。
