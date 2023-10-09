import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.preprocessing import load_SDD
from tqdm import tqdm
import seaborn as sns
import subprocess
import glob
import pdb
from multiprocessing import Pool
import yaml
from darts import TimeSeries
from darts.models.forecasting.random_forest import RandomForest
from darts.models.forecasting.varima import VARIMA
from darts.models.forecasting.arima import ARIMA
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

def forecast_model(m):
    model = VARIMA(p=10,d=1,trend='c')
    verbose = False 
    retrain = True
    show_warnings = False
    mid = mids[m]
    if len(series[m]) <= OBS_LEN + PRED_LEN: # Series is too short
        print("Too short")
        return None
    forecasts = [ fc.pd_dataframe() for fc in model.historical_forecasts(series[m], forecast_horizon=PRED_LEN, train_length=50, last_points_only=False, enable_optimization=True, retrain=retrain, verbose=verbose, show_warnings=show_warnings) ]
    if len(forecasts) == 0: # Series is too short
        print("No forecasts")
        return None
    for j in range(len(forecasts)):
        forecasts[j]['ahead'] = np.arange(PRED_LEN) + 1
        forecasts[j].rename({"x" : "pred_x", "y": "pred_y"}, axis=1, inplace=True)
        forecasts[j] = forecasts[j].reset_index()
        forecasts[j].frame = forecasts[j].frame.min() - 1
    forecasts = pd.concat(forecasts)
    forecasts['metaId'] = mid
    return forecasts

if __name__ == "__main__": 
    scene = sys.argv[1]
    # Setup
    plot_future_samples = False 
    overwrite = False
    savename = './data/darts/' + scene + '.csv'
    OBS_LEN = 10  # in timesteps
    PRED_LEN = 80  # in timesteps
    DOWNSAMPLE_FACTOR_TRAINING = 1
    TRAIN_LEN = (OBS_LEN + PRED_LEN + 1)
    if os.path.exists(savename) and not overwrite:
        sys.exit()

    mode = 'train' if scene in os.listdir('./data/train/') else 'test'
    frames, annotations, _ = load_SDD(scene, mode=mode, load_frames=False)
    annotations.drop(['sceneId'], axis=1, inplace=True)
    annotations.set_index('frame', inplace=True)

    # Get darts series for each agent
    series = []
    mids = []
    noise = 1e-1
    for mid in annotations.metaId.unique():
        anns = annotations[annotations.metaId==mid]
        try:
            curr_anns = annotations[annotations.metaId==mid][['x','y']]
            curr_anns.x = curr_anns.x + np.random.randn(len(curr_anns.x))
            curr_anns.y = curr_anns.y + np.random.randn(len(curr_anns.y))
            series += [ TimeSeries.from_dataframe(curr_anns, value_cols=['x', 'y']) ]
            mids += [ mid ]
        except:
            print(f"Failed for metaId: {mid}")
    prediction_df = Parallel(n_jobs=-1)(delayed(forecast_model)(m) for m in range(len(mids)))
    prediction_df = [df for df in prediction_df if df is not None]
    prediction_df_concatenated = pd.concat(prediction_df, axis=0, ignore_index=True)
    all_data = pd.merge(
        annotations,
        prediction_df_concatenated,
        on=['frame','metaId']
    )
    os.makedirs('./data/darts/', exist_ok=True)
    all_data.to_csv('./data/darts/' + scene + '.csv')
