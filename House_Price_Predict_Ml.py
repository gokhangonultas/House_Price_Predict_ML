#GEREKLİ KÜTÜPHANLER
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate,RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_rows', None)

#VERİNİN OKUTULMASI

train = pd.read_csv("house_prices/train.csv")
test = pd.read_csv("house_prices/test.csv")
df = train.append(test).reset_index(drop=True)
df.head()

#DEĞİŞKEN TÜRLERİNE GÖRE AYIRMA

cat_cols, num_cols, cat_but_car = grab_col_names(df)

#BAĞIMLI DEĞİŞKEN DEĞERLERİ
df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])



#Bağımlı Değişkene göre Korelasyonları Hesaplayacak Yüzde 60 da Büyük ve Küçük
def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df, num_cols)

#DEĞİŞKEN ÜRETME
# Lot Frontage
df["LOTFRONTAGE_RATIO"] = df["LotFrontage"] / df["LotArea"] * 100
df["LOTFRONTAGE_RATIO"].fillna(0, inplace=True)
df.head()

#%%

# Building Age
from datetime import date

todays_date = date.today()
todays_date.year


df["BUILDING_AGE"] = todays_date.year - df["YearBuilt"]
df["BUILDING_AGE_CAT"] = pd.qcut(df["BUILDING_AGE"], 4, labels=["New_house", "Middle_aged", "Middle_Old", "Old"])
df["Sold_Diff"] = df["YrSold"] - df["YearBuilt"]
df["House_Demand"] = pd.qcut(df["Sold_Diff"], 4, labels=["High_Demand", "Normal_Demand", "Less_Demand", "Least_Demand"])
df["BUILDING_AGE"].describe().T
df["Garage_Age"] = df["GarageYrBlt"] - df["YearBuilt"]
df["GARAGE_YEAR_DIFF"] = df["GarageYrBlt"] - df["YearBuilt"]

# First floor ratio
df["FIRST_FLOOR_RATIO"] = df["1stFlrSF"] / df["GrLivArea"] * 100

# Basement Ratio
df[["TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2"]].head(10)
df[df["BsmtFinSF2"] != 0][["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]].head(10)

# Uncomplete ratio:
df["UNCOMP_BSMT_RATIO"] = df["BsmtUnfSF"] / df["TotalBsmtSF"] * 100

# Total bath
df["TOTAL_BATH"] = (df["BsmtHalfBath"] + df["HalfBath"]) * 0.5 + df["BsmtFullBath"] + df["FullBath"]
df["TOTAL_FULL_BATH"] = df["FullBath"] + df["BsmtFullBath"]
df["TOTAL_HALF_BATH"] = df["HalfBath"] + df["BsmtHalfBath"] * 0.5

# Other Rooms
df["NUMBER_OF_OTHER_ROOM"] = df["TotRmsAbvGrd"] - df["KitchenAbvGr"] - df["BedroomAbvGr"]

# Average Room Area
df["AVERAGE_ROOM_AREA"] = df["GrLivArea"] / (df["TotRmsAbvGrd"] + df["TOTAL_BATH"])

# Total porch area
df["TOTAL_PORCH_AREA"] = df["WoodDeckSF"] + df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df[
    "ScreenPorch"]

# Garage ratio
df["GARAGE_RATIO"] = df["GarageArea"] / df["LotArea"] * 100

# Garage Area Per car
df["GARAGE_AREA_PER_CAR"] = df["GarageArea"] / df["GarageCars"]

# Total Garden Area:
df["GARDEN_AREA"] = df["LotArea"] - df["GarageArea"] - df["TOTAL_PORCH_AREA"] - df["TotalBsmtSF"]
df["GARDEN_RATIO"] = df["GARDEN_AREA"] / df["LotArea"] * 100
df[["GARDEN_RATIO", "GARDEN_AREA", "LotArea"]].head(30)
df["LotArea_Cat"] = pd.qcut(df["LotArea"],4,["Small","Medium","Big","Huge"])

#%%

df['HASPOOL'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
df['HAS2NDFLOOR'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
df['HASGARAGE'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
df['HASBSMT'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
df['HASSOMINE'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
df['TotalSF'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] + df['1stFlrSF'] + df['2ndFlrSF'])

# 1. kat ve bodrum metrekare
df["NEW_SF"] = df["1stFlrSF"] + df["TotalBsmtSF"]

# toplam m^2
df["NEW_TOTAL_M^2"] = df["NEW_SF"] + df["2ndFlrSF"]

# Garaj alanı ve metrekarelerin toplamı
df["NEW_SF_G"] = df["NEW_TOTAL_M^2"] + df["GarageArea"]

df['NEW_TOTAL_LVNGAR'] = (df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'])

df["YEAR_OVERALL"] = df["YearRemodAdd"] * df["OverallQual"]
df["NEW_TOTALQUAL"] = df["OverallQual"] * df["GarageArea"] * df["GrLivArea"]
df["YEAR_REMOD"] = df["YearBuilt"] - df["YearRemodAdd"]
df["NEW_AREA"] = df["GrLivArea"] + df["GarageArea"]

#LİSTE BİRLEŞİMİ YAPTIK DEĞİŞKENLERDE
liste = [
    ["MSSubClass", "MSZoning"],
    ["MSSubClass", "BUILDING_AGE_CAT"],
    ["Neighborhood", "HouseStyle"],
    ["HouseStyle", "OverallQual"],
    ["HouseStyle", "OverallCond"],
    ["HouseStyle", "YearRemodAdd"],
    ["HouseStyle", "RoofStyle"],
    ["HouseStyle", "Exterior1st"],
    ["HouseStyle", "MasVnrType"],
    ["SaleType", "SaleCondition", "HouseStyle"],
    ["SaleType", "HouseStyle", "MSSubClass"],
    ["LotConfig", "LotShape"],
    ["LotConfig", "Neighborhood"],
    ["LotArea_Cat", "Neighborhood"],
    ["LandContour", "Neighborhood"]
]

def colon_bros(dataframe, liste):

    for row in liste:
        colon = [col for col in dataframe.columns if col in row]
        dataframe["_".join(map(str, row))] = ["_".join(map(str, i)) for i in dataframe[colon].values]

colon_bros(df,liste)

#RARE ENCODİNG
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = rare_encoder(df, 0.01, cat_cols)

useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]

cat_cols = [col for col in cat_cols if col not in useless_cols]

for col in useless_cols:
    df.drop(col, axis=1, inplace=True)

#Label Encoding & One-Hot Encoding
cat_cols = cat_cols + cat_but_car

df = one_hot_encoder(df, cat_cols, drop_first=False)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]

df[useless_cols_new].head()

for col in useless_cols_new:
    df.drop(col, axis=1, inplace=True)



#İŞE YARAMAYAN DEĞERLERİ TOPLADIK

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df = rare_encoder(df, 0.01, cat_cols)

useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]

cat_cols = [col for col in cat_cols if col not in useless_cols]

for col in useless_cols:
    df.drop(col, axis=1, inplace=True)



#EKSİK DEĞERLERİ KNN ATAMASI YAPARAK DOLDURUYORUZ
missing_values_table(df)
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_filled = imputer.fit_transform(df[na_cols])
df[na_cols] = pd.DataFrame(df_filled, columns=df[na_cols].columns)

missing_values_table(df)




for col in num_cols:
    print(col, check_outlier(df, col, q1=0.01, q3=0.99))
#for col in num_cols:
    #replace_with_thresholds(df, col, q1=0.01, q3=0.99)

#Train-Test Ayrımı
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

# y = train_df["SalePrice"]
y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

#LGMB MODELİ OLUŞTURUYORUZ

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate,RandomizedSearchCV

lgb_model = LGBMRegressor(random_state=17, use_missing=True)

lgb_random_params = {"num_leaves" : np.random.randint(2, 10, 2),
                     "max_depth": np.random.randint(2, 20, 10),
                     "n_estimators": [int(x) for x in np.linspace(start=200, stop=2000, num=50)],
                     "min_child_samples": np.random.randint(5, 20, 10),
                     "reg_alpha": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "reg_lambda": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                     "learning_rate": [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9,1,3,5,7],
                     "colsample_bytree": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
                     "min_child_weight" : [0.001,0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                    "max_bin": np.random.randint(2, 50, 10),
                    'bagging_fraction': [0.01,0.1,0.2,0.3,0.02,0.5,0.7,0.9],
                    'bagging_freq': np.random.randint(2, 10, 5),
                    'min_sum_hessian_in_leaf' : [0.02,0.01]
                     }

lgb_random = RandomizedSearchCV(estimator=lgb_model,param_distributions=lgb_random_params,
                                n_iter=100,
                                cv=3,
                                verbose=True,
                                random_state=42,
                                n_jobs=-1)


lgb_random.fit(X, y)


#FEAUTURES GRAFİĞİ
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(50, 50))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

#FİNAL MODELİ
final_model = lgb_model.set_params(**lgb_random.best_params_).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse

feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': X.columns})
num_summary(feature_imp, "Value", True)
feature_imp[feature_imp["Value"] > 0].shape
feature_imp[feature_imp["Value"] < 1].shape
zero_imp_cols = feature_imp[feature_imp["Value"] < 10]["Feature"].values
selected_cols = [col for col in X.columns if col not in zero_imp_cols]


 #LGBM MODEL EN İYİ PARAMETRELERİ GİRİYORUZ
lgbm_model = LGBMRegressor(random_state=17)

lgbm_params = {"num_leaves" : [8],
               "max_depth": [6],
               "n_estimators": [1779],
               "min_child_samples": [18],
               "reg_alpha": [0.1],
               "reg_lambda": [0.7,0.9],
               "learning_rate": [0.02],
               "colsample_bytree": [0.2],
               "min_child_weight" : [0.02],
               "max_bin": [28],
               'bagging_freq': [6],
               'bagging_fraction': [0.9],
              'min_sum_hessian_in_leaf': [0.00245]
              }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X[selected_cols], y)


final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X[selected_cols], y)
rmse = np.mean(np.sqrt(-cross_val_score(final_model, X[selected_cols], y, cv=5, scoring="neg_mean_squared_error")))
rmse

#SONUÇLARIN YÜKLENMESİ
test_df['Id'] = test_df['Id'].astype('int64')
submission_df = pd.DataFrame()
submission_df['Id'] = test_df["Id"]
y_pred_sub = final_model.predict(test_df[selected_cols])
y_pred_sub = np.expm1(y_pred_sub)
submission_df['SalePrice'] = y_pred_sub
submission_df.to_csv('submission.csv', index=False)