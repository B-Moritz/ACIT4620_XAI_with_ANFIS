import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import copy
import math

from anfis_ba import TSKModel


# Representation? - 2 fold - one array for the sets, one for the consequence parameters

class BeesFitTSK():
    
    def __init__(self, 
                 n_scout_bees, 
                 n_best_sites, 
                 n_elite_sites, 
                 n_workers_best, 
                 n_workers_elite, 
                 n_neighborhod_size,
                 slimit,
                 training_data, test_data, 
                 activate_debuging=False):
        # Initialize the swarm algorithm with a population and the parameters
        self.n_scout_bees = n_scout_bees
        self.n_best_sites = n_best_sites
        self.n_elite_sites = n_elite_sites
        self.n_workers_best = n_workers_best
        self.n_workers_elite = n_workers_elite
        self.n_neighborhod_size = n_neighborhod_size
        self.slimit = slimit
        self.train_data = training_data
        self.test_data = test_data
        self.activate_debuging = activate_debuging

        # Calculate the mutation increment for each feature
        self.mutation_increments = ((self.train_data.iloc[:,1:].max() - self.train_data.iloc[:,1:].min()) / 10).to_dict()
        # Initialize the population
        self.bee_population = np.empty(n_scout_bees,dtype=object)
        
    def initialize_population(self, n_rules=20, n_fuzzy_sets=15, trap_quantile=0.5, expressions={}):
        self.n_rules = n_rules
        self.n_fuzzy_sets = n_fuzzy_sets
        self.expressions = expressions
        self.trap_quantile = trap_quantile
        for i in range(len(self.bee_population)):
            # For each solution in population, initialize model with kmeans
            cur_tsk_model = TSKModel()
            cur_tsk_model.create_rulebase_kmeans(self.train_data, 
                                                 #n_rules=n_rules, 
                                                 n_fuzzy_sets=n_fuzzy_sets, 
                                                 expressions=expressions, 
                                                 trap_quantile=self.trap_quantile)
            
            self.bee_population[i] = cur_tsk_model
            # Evaluate fitness
            self.bee_population[i].calculate_rmse(self.train_data)
        # Sort population on fitness
        self.sort_population()
        self.all_time_best_model = self.bee_population[0]

    def sort_population(self):
        self.bee_population = sorted(self.bee_population, key=lambda x: x.rmse)
        if self.activate_debuging:
            print("Fitness in population")
            for i, site in enumerate(self.bee_population):
                print(f"Position {i}: {site.rmse}")

    def main_loop(self):
        # Select the scouts for elite sites and perform local search
        for i, elite_site in enumerate(self.bee_population[:self.n_elite_sites]):
            if self.activate_debuging: print(f"Work on model {i} (elite)")
            self.bee_population[i] = self.site_exploitation(elite_site, self.n_workers_elite, self.n_neighborhod_size)

        for i, remaining_best_site in enumerate(self.bee_population[self.n_elite_sites:self.n_best_sites]):
            if self.activate_debuging: print(f"Work on model {self.n_elite_sites + i} (remaining best)")
            self.bee_population[self.n_elite_sites + i] = self.site_exploitation(remaining_best_site, self.n_workers_best, self.n_neighborhod_size)

        for i, abandoned_site in enumerate(self.bee_population[self.n_best_sites:]):
            if self.activate_debuging: print(f"Work on model {self.n_best_sites + i} (abandoned)")
            cur_tsk_model = TSKModel()
            cur_tsk_model.create_rulebase_kmeans(self.train_data, 
                                                 #n_rules=self.n_rules, 
                                                 n_fuzzy_sets=self.n_fuzzy_sets, 
                                                 expressions=self.expressions, 
                                                 trap_quantile=self.trap_quantile)
            cur_tsk_model.calculate_rmse(self.train_data)
            self.bee_population[self.n_best_sites + i] = cur_tsk_model

        self.sort_population()
        # Check if there is a new best site
        if self.all_time_best_model.rmse > self.bee_population[0].rmse:
            self.all_time_best_model = copy.deepcopy(self.bee_population[0])

        print(f"Current best fitness: {self.all_time_best_model.rmse:.2f}")
        return self.all_time_best_model.rmse

    def update_set_param(self, old_param_val, increment, previous_parameter, next_parameter):
        new_param_value = old_param_val + np.random.normal(loc=0, scale=1)*increment
        if new_param_value < previous_parameter:
            return previous_parameter
        elif new_param_value > next_parameter:
            return next_parameter
        else:
            return new_param_value

    def update_consequent_param(self, old_param, shrinking_factor):
        return old_param + np.random.normal(0, 1)*old_param*shrinking_factor

    def site_exploitation(self, site : TSKModel, n_followers, n_changes:int):
        cur_best = site
        improvement_falg = False
        n_changes = math.ceil(n_changes * site.get_training_counter() / self.slimit)
        shrinking_factor = 1 - (site.get_training_counter()/self.slimit)
        set_param_size = len(site._feature_fuzzy_sets)
        for worker in range(n_followers):
            # Apply change to the current fuzzy sets
            cur_copy:TSKModel = copy.deepcopy(site)
            for neighborhod_change in range(n_changes):
                # Iterate over the number of changes that should be made to the site
                if np.random.randint(2):
                    # Change consequent of anticedent
                    selected_rules = np.random.randint(len(site.rulebase.feature_names))
                    selected_param = np.random.randint(len(cur_copy.rulebase.rules[selected_rules].consequent.params_list)+1)
                    
                    if selected_param == len(cur_copy.rulebase.rules[selected_rules].consequent.params_list):
                        old_param = cur_copy.rulebase.rules[selected_rules].consequent.const
                        cur_copy.rulebase.rules[selected_rules].consequent.const = self.update_consequent_param(old_param, shrinking_factor)
                    else:
                        old_param = cur_copy.rulebase.rules[selected_rules].consequent.params_list[selected_param]
                        cur_copy.rulebase.rules[selected_rules].consequent.params_list[selected_param] = self.update_consequent_param(old_param, shrinking_factor)
                else:
                    set_to_modify = np.random.randint(set_param_size)
                    subset_to_modify = np.random.randint(len(cur_copy._feature_fuzzy_sets[set_to_modify].fuzzy_sets))
                    param_to_modify = np.random.randint(4)
                    # Get the current param list of the selected set
                    cur_param_list = cur_copy._feature_fuzzy_sets[set_to_modify].fuzzy_sets[subset_to_modify].get_param_list()
                    # Get the selected param value
                    cur_selected_param = cur_param_list[param_to_modify]
                    if param_to_modify == 0:
                        next_param = cur_param_list[param_to_modify+1]
                        previous_param = -np.inf
                    elif param_to_modify == 3:
                        previous_param = cur_param_list[param_to_modify-1]
                        next_param = np.inf
                    else:
                        next_param = cur_param_list[param_to_modify+1]
                        previous_param = cur_param_list[param_to_modify-1]

                    # Get the current value of the parameter to be changed
                    cur_feature_name = cur_copy._feature_fuzzy_sets[set_to_modify]._feature_name
                    cur_param_list[param_to_modify] = self.update_set_param(
                                                                                cur_selected_param, 
                                                                                self.mutation_increments[cur_feature_name], 
                                                                                previous_param, 
                                                                                next_param
                                                                            )
                    
                    cur_copy._feature_fuzzy_sets[set_to_modify].fuzzy_sets[subset_to_modify].set_param_list(cur_param_list)

                
            # Calculate fitness of the worker
            cur_fitness = cur_copy.calculate_rmse(self.train_data)
            if cur_fitness < self.all_time_best_model.rmse and self.activate_debuging:
                if self.activate_debuging: print("Found fitness " + str(cur_fitness) + "While site rmse is " + str(site.rmse))
            if cur_fitness < site.rmse:
                cur_best = cur_copy
                improvement_falg = True

        if not improvement_falg:
            cur_best.increment_training_counter()

        if not improvement_falg and self.slimit <= cur_best.get_training_counter():
            # abandon the site if it has not been improved for n number of generations
            if self.activate_debuging: print("Abandoning site")
            cur_tsk_model = TSKModel()
            cur_tsk_model.create_rulebase_kmeans(self.train_data, 
                                                 #n_rules=self.n_rules, 
                                                 n_fuzzy_sets=self.n_fuzzy_sets, 
                                                 expressions=self.expressions, 
                                                 trap_quantile=self.trap_quantile)
            
            cur_tsk_model.calculate_rmse(self.test_data)
            return cur_tsk_model
        
        return cur_best

                

if __name__=="__main__":
    expressions = {"P":"Precipitation", "E":"Potential evapotranspiration", "PB":"Precipation balance", "Tave":"Tave"}
    selected_features = ["Relative_yield_change", "Tave", "Tmax", "Tmin"]
    train_data = pd.read_csv("dataset/matlab_1_train.csv")[selected_features]
    test_data = pd.read_csv("dataset/matlab_1_test.csv")[selected_features]
    training_model:BeesFitTSK = BeesFitTSK(8, 5, 3, 20, 50, 10, 1000, train_data, test_data, activate_debuging=True)
    training_model.initialize_population(10, 3, trap_quantile=0.4)
    
    for i in range (10000):
        training_model.main_loop()
        

            
            




