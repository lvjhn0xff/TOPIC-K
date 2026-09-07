# Datasets 
from core.datasets.custom import load_custom
from core.datasets.openml import load_openml
from core.datasets.suites import load_cc18
from core.datasets.suites import load_ctr23 
from core.datasets.two_dims import load_moons
from core.datasets.two_dims import load_circles 
from core.datasets.two_dims import load_xor
from core.datasets.two_dims import load_spirals 
from core.datasets.two_dims import load_checkerboard 
from core.models.classifiers import *
from core.models.regressors import *


from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Tuned NCVS (Nested Cross Validation Set-Up) 
from core.experiments.tuned_ncvs import TunedNCVS

from core.topick import TOPICClassifier

# Run configuration.
RANDOM_STATE = 42

# Load dataset. 
print(":: Loading dataset.")
# X, y, make_pipeline = load_moons(
#     n_samples=1000, 
#     noise=0.01, 
#     random_state=RANDOM_STATE
# )
X, y, make_model = load_openml(
    "sonar"
)

# Tuned NCVS object. 
tuned_ncvs = TunedNCVS(
    "classification", 
    X, y, 
    make_model,
    random_state=20
)

tuned_ncvs.define_model = lambda trial:  TOPICClassifier(
    M=1, 
    K=5, 
    n_particles=30, 
    n_swarms=3, 
    max_iter=1000
)

# Execute Tuned NCVS object.
tuned_ncvs.run()