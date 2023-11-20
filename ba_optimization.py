import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from anfis_ba import TSKModel


# Representation? - 2 fold - one array for the sets, one for the consequence parameters

class BeesFitTSK():

    def __init__(self, 
                 n_scout_bees, 
                 n_best_sites, 
                 n_elite_sites, 
                 n_workers_best, 
                 n_workers_elite, 
                 neightbourhod_search,
                 training_data, test_data):
        # Initialize the swarm algorithm with a population and the parameters
        self.n_scout_bees = n_scout_bees
        self.n_best_sites = n_best_sites
        self.n_elite_sites = n_elite_sites
        self.n_workers_best = n_workers_best
        self.n_workers_elite = n_workers_elite
        self.neightbourhod_search = neightbourhod_search
        self.training_data = training_data
        self.tset_data = test_data
        # Initialize the population
        self.bee_population = []
        
    def initialize_population(self, n_rules=20, n_fuzzy_sets=15, expressions={}):
        for i in range(len(self.bee_population)):
            cur_tsk_model = TSKModel()
            cur_tsk_model.create_rulebase_kmeans(self.train_data, 
                                                 n_rules=n_rules, 
                                                 n_fuzzy_sets=n_fuzzy_sets, 
                                                 expressions=expressions, 
                                                 inner_bound_factor=np.random.uniform())
            
            self.bee_population.append(cur_tsk_model)
            self.bee_population[i].calculate_rmse(self.test_data)

        self.sort_population()

    def sort_population(self):
        sorted(self.bee_population, key=lambda x: x.rmse)

    def main_loop(self):
        self.all_time_best_model = None
        # Select the scouts for elite sites and perform local search
        for elite_site in self.bee_population[:self.n_elite_sites]:
            

    def elite_site_exploitation(self, site):
        set_param_size = len(site._feature_fuzzy_sets)
        for worker in self.n_workers_elite:
            # Apply change to the current fuzzy sets
            set_to_modify = np.random.randint(set_param_size)
            param_to_modify = np.random.randint(4)

            """The current status is that we need to determine a good mutation for a local search. how to make proportanate changes to the sets"""
            

        

            
            




