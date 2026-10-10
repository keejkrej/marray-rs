#!/usr/bin/env python3
"""
Generates Python reference golden results for mlab-rs fidelity verification.
Computes ground-truth outputs from numpy, scipy, scikit-learn, scikit-image, and numpy-ml.
"""

import json
from pathlib import Path
import numpy as np
import scipy.linalg as spla
import scipy.special as spspec
import scipy.stats as spstats
import scipy.signal as spsig
import scipy.optimize as spopt
import scipy.integrate as spinteg
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sklearn.metrics as skm
import skimage.color as skcolor
import skimage.filters as skfilt
import skimage.metrics as skmetrics
from numpy_ml.linear_models import RidgeRegression as NmlRidge

OUT_FILE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "python_reference.json"


def generate_all():
    fixtures = {}

    # 1. NumPy References
    np_ref = {}
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np_ref["linspace"] = np.linspace(0.0, 1.0, 5).tolist()
    np_ref["logspace"] = np.logspace(0.0, 3.0, 4, base=10.0).tolist()
    np_ref["geomspace"] = np.geomspace(1.0, 1000.0, 4).tolist()
    np_ref["cumsum"] = np.cumsum(x).tolist()
    np_ref["diff"] = np.diff(x).tolist()
    np_ref["percentile_25"] = float(np.percentile(x, 25))
    np_ref["percentile_50"] = float(np.percentile(x, 50))
    np_ref["percentile_75"] = float(np.percentile(x, 75))

    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    np_ref["cov"] = np.cov(m).tolist()
    np_ref["corrcoef"] = np.corrcoef(m).tolist()
    np_ref["inv"] = np.linalg.inv(m).tolist()
    np_ref["det"] = float(np.linalg.det(m))
    np_ref["norm_l2"] = float(np.linalg.norm(x))

    fixtures["numpy"] = np_ref

    # 2. SciPy References
    sp_ref = {}
    sp_ref["gamma_5"] = float(spspec.gamma(5.0))
    sp_ref["gamma_half"] = float(spspec.gamma(0.5))
    sp_ref["gammaln_5"] = float(spspec.gammaln(5.0))
    sp_ref["beta_2_3"] = float(spspec.beta(2.0, 3.0))
    sp_ref["digamma_1"] = float(spspec.digamma(1.0))
    sp_ref["erf_half"] = float(spspec.erf(0.5))
    sp_ref["erfinv_half"] = float(spspec.erfinv(0.5))
    sp_ref["expit_0"] = float(spspec.expit(0.0))
    sp_ref["logit_half"] = float(spspec.logit(0.5))
    sp_ref["logsumexp"] = float(spspec.logsumexp([1.0, 2.0, 3.0]))
    sp_ref["softmax"] = spspec.softmax([1.0, 2.0, 3.0]).tolist()
    sp_ref["jv_0_24"] = float(spspec.jv(0, 2.4048255577))
    sp_ref["iv_0_1"] = float(spspec.iv(0, 1.0))

    # Stats
    sp_ref["norm_cdf_0"] = float(spstats.norm.cdf(0.0))
    sp_ref["norm_ppf_05"] = float(spstats.norm.ppf(0.5))
    sp_ref["t_cdf_0"] = float(spstats.t.cdf(0.0, df=10))
    sp_ref["ttest_1samp_stat"], sp_ref["ttest_1samp_pval"] = [float(v) for v in spstats.ttest_1samp([1.0, 2.0, 3.0, 4.0, 5.0], 3.0)]
    sp_ref["wasserstein_distance"] = float(spstats.wasserstein_distance([0.0, 1.0], [1.0, 2.0]))

    # Signal & Optimize & Integrate
    b, a = spsig.butter(2, 0.2, btype="low", analog=False)
    sp_ref["butter_b"] = b.tolist()
    sp_ref["butter_a"] = a.tolist()
    sp_ref["brentq_root"] = float(spopt.brentq(lambda z: z*z - 4.0, 0.0, 3.0))
    val, err = spinteg.quad(lambda z: np.sin(z), 0.0, np.pi)
    sp_ref["quad_val"] = float(val)
    sp_ref["quad_err"] = float(err)

    fixtures["scipy"] = sp_ref

    # 3. Scikit-Learn References
    skl_ref = {}
    x_skl = np.array([[1.0, -1.0, 2.0], [2.0, 0.0, 0.0], [0.0, 1.0, -1.0]])
    scaler = StandardScaler().fit(x_skl)
    skl_ref["scaler_mean"] = scaler.mean_.tolist()
    skl_ref["scaler_var"] = scaler.var_.tolist()
    skl_ref["scaler_transform"] = scaler.transform(x_skl).tolist()

    x_reg = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_reg = np.array([2.0, 4.0, 6.0, 8.0])
    lr = LinearRegression().fit(x_reg, y_reg)
    skl_ref["lr_coef"] = lr.coef_.tolist()
    skl_ref["lr_intercept"] = float(lr.intercept_)
    skl_ref["lr_pred"] = lr.predict([[5.0]]).tolist()

    ridge = Ridge(alpha=1.0).fit(x_reg, y_reg)
    skl_ref["ridge_coef"] = ridge.coef_.tolist()
    skl_ref["ridge_intercept"] = float(ridge.intercept_)
    skl_ref["ridge_pred"] = ridge.predict([[5.0]]).tolist()

    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    skl_ref["accuracy"] = float(skm.accuracy_score(y_true, y_pred))
    skl_ref["precision"] = float(skm.precision_score(y_true, y_pred))
    skl_ref["recall"] = float(skm.recall_score(y_true, y_pred))
    skl_ref["f1"] = float(skm.f1_score(y_true, y_pred))

    fixtures["sklearn"] = skl_ref

    # 4. Scikit-Image References
    skimg_ref = {}
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[0, 0, 0] = 255
    gray = skcolor.rgb2gray(rgb)
    skimg_ref["rgb2gray_0_0"] = float(gray[0, 0])

    rgb_f = np.zeros((1, 1, 3), dtype=float)
    rgb_f[0, 0, 0] = 1.0
    hsv = skcolor.rgb2hsv(rgb_f)
    skimg_ref["rgb2hsv_val"] = hsv[0, 0].tolist()

    bimodal = np.zeros((16, 16), dtype=np.uint8)
    bimodal[0:8, :] = 50
    bimodal[8:16, :] = 200
    skimg_ref["otsu_thresh"] = float(skfilt.threshold_otsu(bimodal))

    fixtures["skimage"] = skimg_ref

    # 5. Numpy-ML References
    nml_ref = {}
    r_nml = NmlRidge(alpha=1.0, fit_intercept=True)
    r_nml.fit(x_reg, y_reg)
    nml_ref["ridge_beta"] = r_nml.beta.tolist()
    nml_ref["ridge_pred"] = r_nml.predict(np.array([[5.0]])).tolist()

    fixtures["numpy_ml"] = nml_ref

    # Save to JSON
    with open(OUT_FILE, "w") as f:
        json.dump(fixtures, f, indent=2)

    print(f"Generated {OUT_FILE} successfully with {len(fixtures)} top-level sections.")


if __name__ == "__main__":
    generate_all()
