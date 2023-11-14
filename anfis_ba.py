"This file contains the program for optimizing ANFIS with the bees algorithm"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


def t_norm_product(m1,m2):
    return m1*m2

class MfTrap():

    def __init__(self, a, b, c, d, name):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.name = name

    def get_membership_degree(self, x):
        if self.a == -np.infty and x < self.c:
            return 1
        if self.c == np.infty and x > self.b:
            return 1

        if x < self.a:
            return 0
        elif x < self.b:
            return (x-self.a)/(self.b-self.a)
        elif x < self.c:
            return 1
        elif x <= self.d:
            return (self.d - x)/(self.d - self.c)
        else:
            return 0
        
    def get_param_list(self):
        return [self.a, self.b, self.c, self.d]
        
class FeatureFuzzySets():
    "This object represents all input features and their fuzzy sets" 
    def __init__(self, set_list, feature_name="Unknown"):        
        self.fuzzy_sets : [MfTrap] = np.array(set_list)
        self._feature_name = feature_name

    def __setitem__(self, index, new_set):
        self.fuzzy_sets[index] = new_set
    
    def __getitem__(self, index):
        return self.fuzzy_sets[index]
        
    def remove_set(self, index):
        self.fuzzy_sets[index] = None
        return True
    
class TKSConsequence():
    """ A TSK consequence of first order with three parameters
        f(x1, x2) = x1 * p + x2 * q + r
    """ 
    def __init__(self, params, const):
        self.params_list = params # Corresponds to the coefficients of input values of the rule 
        self.const = const # this coresponds to r
    
    def calculate_consequence(self, x_list, feature_indexes):
        cur_sum = self.const
        for counter, x_ind in enumerate(feature_indexes):
            cur_sum += x_list[x_ind]*self.params_list[counter]

        return cur_sum


class TSKAntecedent():
    def __init__(self, fuzzy_sets_indexes : []):
        self.fuzzy_sets_indexes : [] = fuzzy_sets_indexes
        self.w = None
        
    def calculate_firing_strenght(self, x_list:[], fuzzy_sets:[FeatureFuzzySets]):
        w = 1
        for feature_number, set_number in self.fuzzy_sets_indexes:
            w = w * fuzzy_sets[feature_number][set_number].get_membership_degree(x_list[feature_number])
        
        self.w = w
        return self.w
    
    def get_firing_strenght(self):
        return self.w

class TSKRule():
    def __init__(self, fuzzy_sets_indexes, params : [float], const : float):
        self.antecedent = TSKAntecedent(fuzzy_sets_indexes)
        self.consequent = TKSConsequence(params, const)
        self.feature_indexes = fuzzy_sets_indexes
        
    def calculate_consequence_func(self, x_list):
        return self.consequent.calculate_consequence(x_list, self.feature_indexes[:, 0])
    
    def to_string(self, fuzzy_sets) -> str:
        antecedent_str = ""
        for i, feature in enumerate(self.feature_indexes):
            antecedent_str += "x" + str(feature[0]) + " in " + fuzzy_sets[feature[0]][feature[1]].name
            if i < (len(self.feature_indexes)-1):
                antecedent_str += " and "

        consequent_str = "y = "
        for i, feature in enumerate(self.feature_indexes[:,0]):
            consequent_str += "x" + str(feature) + "*p" + str(feature)
            if i < (len(self.feature_indexes)-1):
                consequent_str += " + "

        return f"IF {antecedent_str} THEN {consequent_str} + r"
    

class TSKRuleBase():
    def __init__(self, feature_names : [str]):
        self.rules : [TSKRule] = []
        self.feature_names = feature_names

    def appendRule(self, new_rule : TSKRule):
        self.rules.append(new_rule)

    def removeRule(self, index : int):
        self.rules.pop(index)
    
    def print_rulebase(self, fuzzy_sets):
        for rule in self.rules:
            print(rule.to_string(fuzzy_sets))
        
    def write_to_csv(self):
        """This method writes the rule base to csv
        """
        rows = []
        columns = [np.concatenate(self.feature_names, ["p"+i for i in range(len(self.feature_names))], ["r"])]
        rule : TSKRule
        for rule in self.rules:
            rows.append([np.concatenate(rule.feature_indexes[:,1], rule.consequent.params_list, [rule.consequent.const])])

        self.rulebase_df = pd.DataFrame(rows, columns=columns)
        self.rulebase_df.to_csv()


    
class TSKModel():

    def __init__(self):
        self._feature_fuzzy_sets = []
        
    
    def randomize_model(self, n_fuzzy_sets : int, n_fuzzy_rules : int, feature_boudaries : {}, feature_names : [], max_expressions : int=2):
        if n_fuzzy_sets < 2:
            # Too few fuzzy sets
            raise Exception(f"Too few fuzzy sets: n_fuzzy_sets < {2}")
        # Model attributes
        # Numbe rof fuzzy sets per feature
        self.n_fuzzy_sets : int = n_fuzzy_sets
        # Number of rules (should be n_fuzzy_sets**n_features for grid partition)
        self.n_fuzzy_rules : int = n_fuzzy_rules
        # A list of feature names
        self.feature_names = feature_names
        # Feature max min list
        self.feature_boundaries = feature_boudaries
        # Maximum number of expressions in the rules (should be same as number of independent features)
        self.max_expressions = max_expressions

        # Create the random fuzzy sets
        overlap_const = 0.5 # 1 full overlapping between sets, 0 there cannot be anny overlaps
        for feature_number, boundaries in self.feature_boundaries.items():
            cur_feature_sets = []
            portion_size = (boundaries[1] - boundaries[0]) / self.n_fuzzy_sets
            for set_num in range(self.n_fuzzy_sets):
                if set_num == 0:
                    # Make sure the first set is defined to -infinity
                    cur_params = np.sort(np.random.uniform(boundaries[0] + portion_size*(set_num - overlap_const), boundaries[0] + portion_size*(set_num + 1 + overlap_const), size=2))
                    cur_feature_sets.append(MfTrap(-np.infty,-np.infty,cur_params[0],cur_params[1], f"{self.feature_names[feature_number]}{set_num}"))
                elif set_num == (self.n_fuzzy_sets-1):
                    cur_params = np.sort(np.random.uniform(boundaries[0] + portion_size*(set_num - overlap_const), boundaries[0] + portion_size*(set_num + 1 + overlap_const), size=2))
                    cur_feature_sets.append(MfTrap(cur_params[0],cur_params[1], np.infty, np.infty, f"{self.feature_names[feature_number]}{set_num}"))
                else:
                    cur_params = np.sort(np.random.uniform(boundaries[0] + portion_size*(set_num - overlap_const), boundaries[0] + portion_size*(set_num + 1 + overlap_const), size=4))
                    cur_feature_sets.append(MfTrap(cur_params[0],cur_params[1],cur_params[2],cur_params[3], f"{self.feature_names[feature_number]}{set_num}"))

            # Add the sets to the input feature
            self._feature_fuzzy_sets.append(FeatureFuzzySets(cur_feature_sets, self.feature_names[feature_number]))

        feature_list = np.array(list(self.feature_boundaries.keys()))
        self.rulebase : TSKRuleBase = TSKRuleBase(self.feature_names)
        for rule_n in range(self.n_fuzzy_rules):
            # For each rule created 

            # pick some of the input features for the rule
            feature_indexes = np.random.choice(len(feature_list), self.max_expressions, replace=False)
            cur_MF_list = [] # The list containing the index of the mfs for the rule
            params = []
            for feature in feature_indexes:
                # For each feature, pick one of the sets
                cur_set_number = np.random.choice(self.n_fuzzy_sets, 1)[0]
                cur_MF_list.append([feature, cur_set_number])
                params.append(1)
            
            self.rulebase.appendRule(TSKRule(np.array(cur_MF_list), params, 1))

    def calculate_output(self, x_list):
        a_out = 0
        w_total = 0
        rule : TSKRule = None
        for rule in self.rulebase.rules:
            # Calculate the total firing strenght for normalization
            w_total += rule.antecedent.calculate_firing_strenght(x_list, self._feature_fuzzy_sets)
        if w_total == 0:
            # The output is 0 if there is no firing strength
            print(x_list)
            return 0
        
        for rule in self.rulebase.rules:
            cur_firing_strenght = rule.antecedent.get_firing_strenght()
            a_out += (cur_firing_strenght / w_total)*rule.calculate_consequence_func(x_list)

        return a_out
        
    def set_rulebase(self, rulebase : TSKRuleBase):
        # Check that the number of rules and max expressions match the previous configuration
        if len(rulebase.rules) > self.n_fuzzy_rules:
            raise Exception("The number of rules are not matching the previous configuration.")
        if len(rulebase.rules[0].antecedent.fuzzy_sets_indexes) > self.max_expressions:
            raise Exception("The number of expressions per rules are too high.")
        
        self.rulebase = rulebase
        self.rulebase.print_rulebase(self._feature_fuzzy_sets)
        

    def set_feature_fuzzy_sets(self, feature_fuzzy_sets : [FeatureFuzzySets]):
        # Check that there are enough sets
        if len(feature_fuzzy_sets) != len(self.feature_names):
            raise Exception("The number of features are not matching with the existing feature set.")
        if len(feature_fuzzy_sets[0].fuzzy_sets) != self.n_fuzzy_sets:
            raise Exception("The number of sets per feature is not matching existing configuration.")
        
        self._feature_fuzzy_sets = feature_fuzzy_sets

    def create_rulebase_kmeans(self, 
                               train_data : pd.DataFrame, 
                               n_fuzzy_sets : int=3, 
                               inner_bound_factor : float=0.3) -> None:
        """This method derives the TSK rulebase by performing K-Means clustering and Multi-Linear Regression. 


        Parameters
        ----------
        train_data : pandas.DataFrame
            The training data. Dependent variable is column 1 (index 0). Independent features follow
        n_fuzzy_sets : int, optional
            Number of fuzzy sets per independent feature, by default 3
        inner_bound_factor : int, optional
            Factor for setting the core of the trapezoidals. center +/- outer_bound_factor*std, by default 1
        """
        X = train_data.to_numpy()
        cluster_obj = KMeans(n_clusters=10, n_init=10)
        cluster_fit = cluster_obj.fit(X)
        merged_data = pd.merge(train_data.iloc[:, 1:], pd.DataFrame({"Subcluster_number" : cluster_fit.labels_}), left_index=True, right_index=True)
        self.n_fuzzy_sets = n_fuzzy_sets
        self.feature_names = merged_data.columns[1:-1]
        self.target_name = merged_data.columns[0]
        self.max_expressions = len(self.feature_names)
        # Consequences are calculated
        subclusters = merged_data["Subcluster_number"].unique()
        r2_scores : [float] = []
        consequent_params : [float] = []
        intercepts : [float] = []
        for subcluster in subclusters:
            cur_X = merged_data[merged_data["Subcluster_number"] == subcluster].iloc[:, 1:-1]
            cur_Y = merged_data[merged_data["Subcluster_number"] == subcluster].iloc[:, 0]
            # Make the fit
            cur_linear = LinearRegression().fit(cur_X, cur_Y)
            # Store coefficients
            consequent_params.append(cur_linear.coef_)
            # Store the constant
            intercepts.append(cur_linear.intercept_)
            # Store the r2 score of the fit
            r2_scores.append(cur_linear.score(cur_X, cur_Y))

        self.r2_scores = r2_scores

        # Derive the membership functions from the clusters (trapeziums)
        cluster_center : [float] = []
        mfs : [FeatureFuzzySets] = []
        # Initialization of the rule list dict
        clusters : {} = {i : [] for i in np.unique(cluster_fit.labels_)}

        for k in range(1, len(merged_data.iloc[:, 1:-1].columns)+1):
            raw_mfs = cluster_fit.cluster_centers_[:,k].reshape(-1, 1)
            # Reduce amount of mfs by clustering in one dimension
            reduced_mfs_middle = KMeans(n_clusters=n_fuzzy_sets, n_init=5).fit(raw_mfs)
            # Storing cluster centers of the current feature
            cluster_center.append(reduced_mfs_middle.cluster_centers_.ravel())
            subcluster_in_clusters = [[] for i in np.unique(reduced_mfs_middle.labels_)]
            for subcluster_number in range(len(raw_mfs)):
                cur_cluster_number = reduced_mfs_middle.labels_[subcluster_number]
                # Add the feature set to the rule list dictionary
                clusters[subcluster_number].append(cur_cluster_number)
                # Keep list of which subcluster are in the root clusters for the feature
                subcluster_in_clusters[cur_cluster_number].append(subcluster_number)
            
            cur_mfs = []
            for cluster, subclusters in enumerate(subcluster_in_clusters):
                # For each cluster in subcluster calculate std for current feature
                cur_std = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].std()
                cur_min = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].min()
                cur_max = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].max()
                cur_center = cluster_center[k-1][cluster]
                cur_mfs.append(MfTrap(cur_center - (abs(cur_center - cur_min)), 
                                      cur_center - cur_std*inner_bound_factor, 
                                      cur_center + cur_std*inner_bound_factor, 
                                      cur_center + (abs(cur_center - cur_max)), 
                                      ""))

            # Add the feature collection of sets
            mfs.append(FeatureFuzzySets(cur_mfs, merged_data.columns[k]))
        # Add the fuzzy sets to model
        self.set_feature_fuzzy_sets(mfs)

        self.n_fuzzy_rules = len(clusters.keys())
        new_rulebase = TSKRuleBase(self.feature_names)
        # Create rulebase
        for rule_num, rule in clusters.items():
            new_rulebase.appendRule(TSKRule(np.array([np.arange(len(rule)), rule]).transpose(), 
                                    consequent_params[rule_num], 
                                    intercepts[rule_num])
                                    )
        # Add the rulebase 
        self.set_rulebase(new_rulebase)



    def show_fuzzy_sets(self):
        """This method is used to plot the fuzzy sets for each feature"""
        fig, axes = plt.subplots(len(self.feature_names), 1, figsize=(10, 10))
        if len(self.feature_names) == 1:
            axes = [axes]

        fig.tight_layout(pad=5)
        for id, feature in enumerate(self.feature_names):
            axes[id].set_title(f"fuzzy sets for {feature}")
            axes[id].set_xlabel(feature)
            axes[id].set_ylabel("membership degree")
            for cur_set in self._feature_fuzzy_sets[id].fuzzy_sets:
                cur_param_list = cur_set.get_param_list()

                if cur_param_list[0] == -np.infty:
                    # Add display boundaries
                    cur_param_list[0] = cur_param_list[2] - 2
                    cur_param_list[1] = cur_param_list[2] - 1

                if cur_param_list[2] == np.infty:
                    # Add display boundaries
                    cur_param_list[2] = cur_param_list[1] + 2
                    cur_param_list[3] = cur_param_list[1] + 1

                deg_list = []
                for param in cur_param_list:
                    deg_list.append(cur_set.get_membership_degree(param))
                
                axes[id].plot(cur_param_list, deg_list)
                
        plt.show()


if __name__ == '__main__':
    # Test features
    test_dict = {0 : (0, 6), 1 : (0, 6)}
    test_names = ["A", "B"]
    test_tsk_model = TSKModel()
    test_tsk_model.randomize_model(4, 4*4, test_dict, test_names, max_expressions=2)

    cur_feature_sets_A = []
    cur_feature_sets_A.append(MfTrap(-np.infty, -np.infty, 0, 1, f"Very small"))
    cur_feature_sets_A.append(MfTrap(1, 2, 2, 3, f"Small"))
    cur_feature_sets_A.append(MfTrap(3, 4, 4, 5, f"Large"))
    cur_feature_sets_A.append(MfTrap(5, 6, np.infty, np.infty, f"Very large"))

    cur_feature_sets_B = []
    cur_feature_sets_B.append(MfTrap(-np.infty, -np.infty, 0, 1, f"Very small"))
    cur_feature_sets_B.append(MfTrap(1, 2, 2, 3, f"Small"))
    cur_feature_sets_B.append(MfTrap(3, 4, 4, 5, f"Large"))
    cur_feature_sets_B.append(MfTrap(5, 6, np.infty, np.infty, f"Very large"))

    feature_fuzzy_sets = {0 : FeatureFuzzySets(cur_feature_sets_A, "A"), 1 : FeatureFuzzySets(cur_feature_sets_B, "B")}
    test_tsk_model.set_feature_fuzzy_sets(feature_fuzzy_sets)

    new_rulebase = TSKRuleBase(test_names)
    possible_params = [[1,1],[0,0], [1,1], [0,0]]
    possible_r = [0,2,-2,8]
    for i in range(4):
        for j in range(4):
            cur_params = possible_params[max(i, j)]
            cur_r = possible_r[max(i, j)]
            cur_feature_mapping = np.array([[0, i], [1, j]])
            new_rulebase.appendRule(TSKRule(cur_feature_mapping, cur_params, cur_r))

    #new_rulebase.appendRule(TSKRule(np.array([[0, 0], [1, 0]]), [1, 1], 0))
    #new_rulebase.appendRule(TSKRule(np.array([[0, 1], [1, 1]]), [0, 0], 1))
    #new_rulebase.appendRule(TSKRule(np.array([[0, 2], [1, 2]]), [1, 1], -2))
    #new_rulebase.appendRule(TSKRule(np.array([[0, 3], [1, 3]]), [0, 0], 3))

    test_tsk_model.set_rulebase(new_rulebase)

    #test_tsk_model.show_fuzzy_sets()

    X = np.linspace(0,6, 100)
    Y = np.linspace(0, 6, 100)
    X, Y = np.meshgrid(X, Y)
    X_flat, Y_flat = X.ravel(), Y.ravel()
    Z = np.empty(100*100)
    for i, x, y in zip(np.arange(len(X_flat)), X_flat, Y_flat):
        Z[i] = test_tsk_model.calculate_output([x, y])

    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z.reshape(X.shape), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.show()


