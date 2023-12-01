"This file contains the program for optimizing ANFIS with the bees algorithm"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import pdb


class MfTrap():
    """The membership function used to define a trapezoidal fuzzy set.
    """

    def __init__(self, a, b, c, d, name):
        # The four parameters used to define the function
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        # The name of the set (linguistic expression) is defined as an attribute to the object
        self.name = name
        # The set is validated. Note how the following must be satisfied a < b <= c < d
        self.validate_set()

    def validate_set(self):
        if self.a > self.b:
            print("a:" + str(self.a) + ", b:" + str(self.b))
            raise Exception("Parameter a is larger than b.")
        if self.b > self.c:
            raise Exception("Parameter b is larger than c.")
        if self.b > self.c:
            raise Exception("Parameter c is larger than d.")
        
    def get_membership_degree(self, x):
        """This method produces the membership degree of the value x to the current defined trapezoidal set (a,b,c,d)

        Parameters
        ----------
        x : float

        Returns
        -------
        float in the universeral interval [0,1]
            The membership degree of the input value.
        """
        # Check for cases where the set represents x values for infinity
        if self.a == -np.infty and x < self.c:
            # If a is negative infinity and x less than c, the memberhsip degree must be 1
            return 1
        if self.d == np.infty and x > self.b:
            # If d is infinity and x greater than b, the memberhsip degree must be 1
            return 1

        # The following if test determines where in what section the x value is for the fuzzy set ¨
        # and reutns the corresponding memberhsip functions
        if x < self.a:
            # If x is less than the first parameter, no membership is acounted for the value
            return 0
        elif x < self.b:
            # If the memberhsip degree is greater than or equal to a and less than b, 
            # the x value is in the line with positive slope
            if self.b - self.a == 0:
                # If b and a are equal, the memberhsip degree must be 0. 
                # There is no fuzziness between a and b.
                return 0
            return (x-self.a)/(self.b-self.a)
        elif x < self.c:
            # If x is less than c and greater than or equal to b, 
            # the membership degree is in the core of the set (highest possible memberhsip degree).
            return 1
        elif x <= self.d:
            # If x is less than or equal to d, the memberhsip degree 
            # is defined by the line with negative slope
            if self.d - self.c == 0:
                # If d and c are equal, the memberhsip degree is 0 (no fuzziness)
                return 1
            return (self.d - x)/(self.d - self.c)
        else:
            # For any vlaue greater than d, the degree is 0
            return 0
        
    def get_param_list(self):
        return [self.a, self.b, self.c, self.d]

    def set_param_list(self, param_list):
        self.a = param_list[0]
        self.b = param_list[1]
        self.c = param_list[2]
        self.d = param_list[3]
        self.validate_set()
        
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
        # This method is used to calculate the function value of the TSK consequent
        cur_sum = self.const
        for counter, x_ind in enumerate(feature_indexes):
            cur_sum += x_list[x_ind]*self.params_list[counter]

        return cur_sum

class TSKAntecedent():
    """This class represents the antecedent part of a TSK rule and can calculate the firing strength for the rule.
       The t-norm used is product.
    """
    def __init__(self, fuzzy_sets_indexes : []):
        self.fuzzy_sets_indexes : [] = fuzzy_sets_indexes
        self.w = None
        
    def calculate_firing_strenght(self, x_list:[], fuzzy_sets:[FeatureFuzzySets]):
        # This method calculates the firing strenght of the antecedents
        # Start by setting the firing strenght to the identity value for the product operator
        w = 1
        for feature_number, set_number in self.fuzzy_sets_indexes:
            # For each set in the antecedent, calculate the membership degree and find the product of degree and the firing strength calculated up until now.
            w = w * fuzzy_sets[feature_number][set_number].get_membership_degree(x_list[feature_number])
        
        self.w = w
        return self.w
    
    def get_firing_strenght(self):
        return self.w

class TSKRule():
    """A class used to defining a TSK rule.
    """
    def __init__(self, fuzzy_sets_indexes, params : [float], const : float):
        # The antecedent of the rule
        self.antecedent = TSKAntecedent(fuzzy_sets_indexes)
        # The consequent of the rule
        self.consequent = TKSConsequence(params, const)
        # The indexes of the sets making up the antecedents
        self.feature_indexes = fuzzy_sets_indexes
        
    def calculate_consequence_func(self, x_list):
        return self.consequent.calculate_consequence(x_list, self.feature_indexes[:, 0])
    
    def to_string(self, fuzzy_sets, expressions) -> str:
        antecedent_str = ""
        for i, feature in enumerate(self.feature_indexes):
            antecedent_str += f"{expressions[feature[0]]} in " + fuzzy_sets[feature[0]][feature[1]].name
            if i < (len(self.feature_indexes)-1):
                antecedent_str += " and "

        consequent_str = "y = "
        for i, feature in enumerate(self.feature_indexes[:,0]):
            consequent_str += f"{expressions[feature]} * {self.consequent.params_list[feature]:.4f}"
            if i < (len(self.feature_indexes)-1):
                consequent_str += " + "

        return f"IF {antecedent_str} THEN {consequent_str} + {self.consequent.const:.4f}"
    
class TSKRuleBase():
    """This class represents a TSK rule base. It has the responsibility to add and remove rules.
        It also has the responsibility to print out the rules.
    """
    def __init__(self, feature_names : [str], expressions : {str:str}):
        self.rules : [TSKRule] = []
        # Storing feature names and creating expressions for printout.
        self.feature_names = feature_names
        self.expressions = []
        for i, feature in enumerate(self.feature_names):
            if feature in expressions:
                self.expressions.append(expressions[feature])
            else:
                self.expressions.append("x"+str(i))

    def appendRule(self, new_rule : TSKRule):
        self.rules.append(new_rule)

    def removeRule(self, index : int):
        self.rules.pop(index)
    
    def print_rulebase(self, fuzzy_sets):
        for rule in self.rules:
            print(rule.to_string(fuzzy_sets, self.expressions))
        
    def write_to_csv(self, fileName="Last_rulebase.csv"):
        """This method writes the rule base to csv
        """
        rows = []
        columns = [np.concatenate(self.feature_names, ["p"+i for i in range(len(self.feature_names))], ["r"])]
        rule : TSKRule
        for rule in self.rules:
            rows.append([np.concatenate(rule.feature_indexes[:,1], rule.consequent.params_list, [rule.consequent.const])])

        self.rulebase_df = pd.DataFrame(rows, columns=columns)
        self.rulebase_df.to_csv(fileName)


class TSKModel():

    def __init__(self, debug:bool = False):
        self._feature_fuzzy_sets = [MfTrap]
        self.test_actuals = []
        self.r_squared = None
        self.test_data = None
        self.debug = debug
        self.generation = 0
        
    
    def randomize_model(self, 
                        n_fuzzy_sets : int, 
                        n_fuzzy_rules : int, 
                        feature_boudaries : {}, 
                        feature_names : [], 
                        max_expressions : int=2):
        
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
        self.rulebase : TSKRuleBase = TSKRuleBase(self.feature_names, expressions={})
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

    def calculate_output(self, x_list, activate_debug=False):
        """This method is used to apply the trained model on some input values.

        Parameters
        ----------
        x_list : [int]
            A list of one value for each input variable x

        Returns
        -------
        float
            The predicted value
        """
        a_out = 0
        w_total = 0
        rule : TSKRule = None

        for rule in self.rulebase.rules:
            # Calculate the total firing strenght for normalization
            cur_firing_strenght = rule.antecedent.calculate_firing_strenght(x_list, self._feature_fuzzy_sets)
            w_total += cur_firing_strenght
        if w_total == 0:
            # The output is 0 if there is no firing strength
            #print(x_list)
            return 0
        
        for i, rule in enumerate(self.rulebase.rules):
            cur_firing_strenght = rule.antecedent.get_firing_strenght()
            cur_consequent_value = rule.calculate_consequence_func(x_list)
            if activate_debug:
                # Print out the rule calculation results for transparency
                print(f"Rule {i}: \t Firing strength: {cur_firing_strenght}, Consequent value: {cur_consequent_value}")
            a_out += (cur_firing_strenght / w_total)*cur_consequent_value

        return a_out
        
    def set_rulebase(self, rulebase : TSKRuleBase):
        # Check that the number of rules and max expressions match the previous configuration
        if len(rulebase.rules) > self.n_fuzzy_rules:
            raise Exception("The number of rules are not matching the previous configuration.")
        if len(rulebase.rules[0].antecedent.fuzzy_sets_indexes) > self.max_expressions:
            raise Exception("The number of expressions per rules are too high.")
        
        self.rulebase = rulebase
        if self.debug:
            self.rulebase.print_rulebase(self._feature_fuzzy_sets)
        
    def set_feature_fuzzy_sets(self, feature_fuzzy_sets : [FeatureFuzzySets]):
        # Check that there are enough sets
        if len(feature_fuzzy_sets) != len(self.feature_names):
            raise Exception("The number of features are not matching with the existing feature set.")
        if len(feature_fuzzy_sets[0].fuzzy_sets) != self.n_fuzzy_sets:
            raise Exception("The number of sets per feature is not matching existing configuration.")
        
        self._feature_fuzzy_sets = feature_fuzzy_sets

    def create_rulebase_kmeans_advanced(self,
                               train_data : pd.DataFrame, 
                               n_fuzzy_sets : int=3, 
                               n_rules : int=10,
                               trap_quantile : float=0.4, 
                               expressions={}) -> None:
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
        # Verify that there are enough features in dataframe (feature 1 is the target/dependent variable)
        if len(train_data.columns) < 2:
            raise Exception("The provided input data does not have a valid dimention: dimention < 2")

        # Verfiy the given number of fuzzy sets
        if n_fuzzy_sets < 1:
            raise Exception(f"Cannot create model with a number of fuzzy sets of {n_fuzzy_sets} for each input.")

        self.n_fuzzy_sets = n_fuzzy_sets

        X = train_data.to_numpy()
        cluster_obj = KMeans(n_clusters=n_rules, n_init=10)
        cluster_fit = cluster_obj.fit(X)
        # Add a feature at the end of the dataframe, containing the cluster labels
        merged_data = pd.merge(train_data, pd.DataFrame({"Subcluster_number" : cluster_fit.labels_}), left_index=True, right_index=True)
        self.merged_dataset = merged_data
        # Resulting dataframe has target in first column, independent features in the middle and the cluster labels in the last colummn
        # Extract the featurenames by ignoring the first and last column
        self.feature_names = merged_data.columns[1:-1]
        # Extract the target name
        self.target_name = merged_data.columns[0]
        self.max_expressions = len(self.feature_names)

        # Extracting all unique subclusters/rules
        subclusters = merged_data["Subcluster_number"].unique()

        # Derive the membership functions from the clusters (trapeziums)
        cluster_center : [float] = []
        mfs : [FeatureFuzzySets] = []
        # Initialization of the rule list dict
        clusters : {} = {i : [] for i in np.unique(cluster_fit.labels_)}

        for k in range(1, len(merged_data.columns)-1):
            # Iterate over all features (independent variables), starts at 1 because first column is not an independent variable
            # k - feature index (column index)

            # Get the cluster centers of the subclusters. Note that the cluster_fit.cluster_centers_ matrix has cluster on axis 0 and feature cluster center along axis 1
            raw_mfs = cluster_fit.cluster_centers_[:,k].reshape(-1, 1)
            # Reduce amount of mfs by clustering in one dimension
            reduced_mfs_middle = KMeans(n_clusters=n_fuzzy_sets, n_init=5).fit(raw_mfs)

            # Storing cluster centers of the current feature
            cluster_center.append(np.sort(reduced_mfs_middle.cluster_centers_.ravel()))
            # Preparing list for mapping what rules each set is used in (index is set number, values are lists with rule index)
            subcluster_in_clusters = [[] for i in np.unique(reduced_mfs_middle.labels_)]
            for subcluster_number in range(len(raw_mfs)):
                # Iterate ower the raw_mfs (rule clusters)
                cur_cluster_number = reduced_mfs_middle.labels_[subcluster_number]
                # Add the feature set to the rule list dictionary. 
                # Note how this is reapeated for each feature in the outer loop and in this loop the current feature is added to all rules
                clusters[subcluster_number].append(cur_cluster_number)
                # Keep list of which subcluster are in the root clusters for the feature
                subcluster_in_clusters[cur_cluster_number].append(subcluster_number)

            # Preparing the mfs list            
            cur_mfs = []
            for cluster, subclusters in enumerate(subcluster_in_clusters):
                # For each set derived for this feature, create the fuzzy trapezodial by calculating the quantile
                cur_mean = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].mean()
                cur_quantile_val = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].quantile(trap_quantile)
                dis_from_center = abs(cur_mean - cur_quantile_val)
                cur_min = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].min()
                cur_max = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].max()
                cur_center = cluster_center[k-1][cluster]
                cur_mfs.append(MfTrap(cur_center - (abs(cur_mean - cur_min)),
                                      cur_center - dis_from_center,
                                      cur_center + dis_from_center,
                                      cur_center + (abs(cur_mean - cur_max)),
                                      f"{merged_data.columns[k]}_{cluster}"))

            # Add the feature collection of sets
            mfs.append(FeatureFuzzySets(cur_mfs, merged_data.columns[k]))
        # Add the fuzzy sets to model
        self.set_feature_fuzzy_sets(mfs)

        
        consequent_train_data = []
        
        antecedents = []
        for rule_num, rule in clusters.items():
            cur_index_matrix = np.array([np.arange(len(rule)), rule]).transpose()
            antecedents.append(TSKAntecedent(cur_index_matrix))
        # Create the dataset to fit the consequents with least squeare method
        for row in range(merged_data.shape[0]):
            cur_x_row = merged_data.iloc[row, 1:-1].to_numpy()
            cur_row_expanded = np.array([])
            w_total = 0
            cur_w_list = []
            # Find firing strength for each rule
            for rule in antecedents:
                rule_w = rule.calculate_firing_strenght(cur_x_row, self._feature_fuzzy_sets)
                w_total += rule_w
                cur_w_list.append(float(rule_w))

            if w_total == 0:
                w_total = 1

            for w in cur_w_list:
                cur_firing_normalized = w/w_total
                if len(cur_row_expanded) == 0:
                    cur_row_expanded = np.concatenate(((cur_x_row*cur_firing_normalized), np.array([cur_firing_normalized])), dtype=float)
                else:
                    cur_row_expanded = np.concatenate((cur_row_expanded, (cur_x_row*cur_firing_normalized), np.array([cur_firing_normalized])), dtype=float)
            
            consequent_train_data.append(cur_row_expanded)
                
        cur_Y = merged_data.iloc[:, 0].to_numpy()
        cur_X = consequent_train_data
        # Estimate the consequence parameters
        linear_estimate = LinearRegression(fit_intercept=False, copy_X=True).fit(cur_X, cur_Y)
        self.r2_scores = linear_estimate.score(cur_X, cur_Y)
        # Extract the parameters
        self.extracted_params = linear_estimate.coef_
        # The next lines partition the parameters into the rules. Each rule has parameters and one intercept which is the last parameter (1 + the last index of the rule)
        rule_params = [[] for i in range(len(clusters.keys()))]
        rule_intercepts = np.empty(len(clusters.keys()), dtype=float)
        param_counter = 0
        for rule_num, rule in clusters.items():
            # For each rule iterate over all features in rule variable
            for param_nr in range(len(rule)):
                rule_params[rule_num].append(self.extracted_params[param_counter + param_nr])

            rule_intercepts[rule_num] = self.extracted_params[param_counter + len(rule)]
            param_counter += len(rule) + 1

        self.n_fuzzy_rules = len(clusters.keys())
        new_rulebase = TSKRuleBase(self.feature_names, expressions)
        # Create rulebase
        for rule_num, rule in clusters.items():
            # Creating a matrix containing the mapping between antecedent place and the fuzzy set index
            cur_index_matrix = np.array([np.arange(len(rule)), rule]).transpose()
            # Defining the rule
            new_rulebase.appendRule(TSKRule(cur_index_matrix, 
                                    rule_params[rule_num], 
                                    rule_intercepts[rule_num])
                                    )
        # Add the rulebase 
        self.set_rulebase(new_rulebase)

    def create_rulebase_kmeans(self,
                               train_data : pd.DataFrame, 
                               n_fuzzy_sets : int=3, 
                               trap_quantile : float=0.4, 
                               expressions={}) -> None:
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
        # Verify that there are enough features in dataframe (feature 1 is the target/dependent variable)
        if len(train_data.columns) < 2:
            raise Exception("The provided input data does not have a valid dimention: dimention < 2")

        # Verfiy the given number of fuzzy sets
        if n_fuzzy_sets < 1:
            raise Exception(f"Cannot create model with a number of fuzzy sets of {n_fuzzy_sets} for each input.")

        self.n_fuzzy_sets = n_fuzzy_sets

        X = train_data.to_numpy()
        cluster_obj = KMeans(n_clusters=n_fuzzy_sets, n_init=10)
        cluster_fit = cluster_obj.fit(X)
        # Add a feature at the end of the dataframe, containing the cluster labels
        merged_data = pd.merge(train_data, pd.DataFrame({"Subcluster_number" : cluster_fit.labels_}), left_index=True, right_index=True)
        self.merged_dataset = merged_data
        # Resulting dataframe has target in first column, independent features in the middle and the cluster labels in the last colummn
        # Extract the featurenames by ignoring the first and last column
        self.feature_names = merged_data.columns[1:-1]
        # Extract the target name
        self.target_name = merged_data.columns[0]
        self.max_expressions = len(self.feature_names)

        # Extracting all unique subclusters/rules
        unique_clusters = merged_data["Subcluster_number"].unique()

        # Derive the membership functions from the clusters (trapeziums)
        cluster_center : [float] = []
        mfs : [FeatureFuzzySets] = []
        # Initialization of the rule list dict
        clusters : {} = {i : [] for i in np.unique(cluster_fit.labels_)}

        for k in range(1, len(merged_data.columns)-1):
            # Iterate over all features (independent variables), starts at 1 because first column is not an independent variable
            # k - feature index (column index)

            # Get the cluster centers of the subclusters. Note that the cluster_fit.cluster_centers_ matrix has cluster on axis 0 and feature cluster center along axis 1
            raw_mfs = cluster_fit.cluster_centers_[:,k]
            
            # Preparing list for mapping what rules each set is used in (index is set number, values are lists with rule index)
            subcluster_in_clusters = [[] for i in range(len(raw_mfs))]
            for subcluster_number in range(len(raw_mfs)):
                # Iterate ower the raw_mfs (rule clusters)
                # Add the feature set to the rule list dictionary. 
                # Note how this is reapeated for each feature in the outer loop and in this loop the current feature is added to all rules
                clusters[subcluster_number].append(subcluster_number)
                # Keep list of which subcluster are in the root clusters for the feature
                subcluster_in_clusters[subcluster_number].append(subcluster_number)

            # Preparing the mfs list            
            cur_mfs = []
            for cluster, subclusters in enumerate(subcluster_in_clusters):
                # For each set derived for this feature, create the fuzzy trapezodial by calculating the quantile
                cur_mean = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].mean()
                cur_quantile_val = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].quantile(trap_quantile)
                dis_from_center = abs(cur_mean - cur_quantile_val)
                cur_min = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].min()
                cur_max = merged_data[merged_data["Subcluster_number"].isin(np.unique(subclusters))].iloc[:, k].max()
                cur_center = raw_mfs[cluster]
                cur_mfs.append(MfTrap(cur_center - (abs(cur_center - cur_min)),
                                      cur_center - dis_from_center,
                                      cur_center + dis_from_center,
                                      cur_center + (abs(cur_center - cur_max)),
                                      f"{merged_data.columns[k]}_{cluster}"))

            # Add the feature collection of sets
            mfs.append(FeatureFuzzySets(cur_mfs, merged_data.columns[k]))
        # Add the fuzzy sets to model
        self.set_feature_fuzzy_sets(mfs)

        
        self.r2_scores = []
        self.extracted_params = []
        rule_params = [[] for i in range(len(clusters.keys()))]
        rule_intercepts = np.empty(len(clusters.keys()), dtype=float)
        antecedents = []
        cur_y = merged_data.iloc[:, 0].to_numpy().ravel()
        for rule_num, rule in clusters.items():
            cur_index_matrix = np.array([np.arange(len(rule)), rule]).transpose()
            antecedents.append(TSKAntecedent(cur_index_matrix))
        # Create the dataset to fit the consequents with least squeare method
        for w_rule in range(len(antecedents)):
            # For each rule, calculate the updated dataset
            consequent_train_data = []
            reduced_y = []
            for row in range(merged_data.shape[0]):
                cur_x_row = merged_data.iloc[row, 1:-1].to_numpy()
                w_total = 0
                cur_w_list = []
                # Find firing strength for each rule
                for rule in antecedents:
                    rule_w = rule.calculate_firing_strenght(cur_x_row, self._feature_fuzzy_sets)
                    w_total += rule_w
                    cur_w_list.append(float(rule_w))

                if w_total > 0 and cur_w_list[w_rule] > 0:
                    cur_firing_normalized = cur_w_list[w_rule]/w_total
                    consequent_train_data.append(np.concatenate((cur_x_row*cur_firing_normalized, np.array([cur_firing_normalized]))))
                    reduced_y.append(cur_y[row])
        
            cur_linear_estimate = LinearRegression(fit_intercept=False, copy_X=True).fit(consequent_train_data, np.array(reduced_y).reshape(-1, 1))    
            self.r2_scores.append(cur_linear_estimate.score(consequent_train_data, np.array(reduced_y).reshape(-1, 1)))
            cur_parameters = cur_linear_estimate.coef_.ravel()
            self.extracted_params.append(cur_parameters)
            rule_params[w_rule] = cur_parameters[:-1]
            rule_intercepts[w_rule] = cur_parameters[-1]


        self.n_fuzzy_rules = len(clusters.keys())
        new_rulebase = TSKRuleBase(self.feature_names, expressions)
        # Create rulebase
        for rule_num, rule in clusters.items():
            # Creating a matrix containing the mapping between antecedent place and the fuzzy set index
            cur_index_matrix = np.array([np.arange(len(rule)), rule]).transpose()
            # Defining the rule
            new_rulebase.appendRule(TSKRule(cur_index_matrix, 
                                    rule_params[rule_num], 
                                    rule_intercepts[rule_num])
                                    )
        # Add the rulebase 
        self.set_rulebase(new_rulebase)


    def test_model(self, test_data):
        self.test_data = test_data
        # Calculate the output values
        cur_actuals = np.empty(test_data.shape[0])
        for i, x_vals in test_data.iloc[:, 1:].iterrows():
            cur_actuals[i] = self.calculate_output(x_vals.to_numpy())

        self.test_actuals = cur_actuals

    def calculate_r_squared(self, test_data=None):
        # This method calculates the r-squared: 1 - RSS/TSS
        self.r_squared = 0
        if len(self.test_actuals) == 0 and len(test_data) > 0:
            # If the actuals have not been calculated, apply the
            self.test_data = test_data
            self.test_model(test_data)
        elif len(self.test_actuals) == 0:
            raise Exception("No test data is available, please provide test data!")

        self.test_model(test_data)
        # Total sum of squares (denominator)
        y = self.test_data.iloc[:, 0].to_numpy()
        y_mean = np.mean(self.test_actuals)
        tss = np.sum((y - y_mean)**2)
        rss = np.sum((y - self.test_actuals)**2)

        self.r_squared = 1 - (rss/tss)
        return self.r_squared
    
    def calculate_rmse(self, test_data=pd.DataFrame()):
        # This method calculates the r-squared: 1 - RSS/TSS
        self.rmse = 0
        if len(self.test_actuals) == 0 and len(test_data) > 0:
            # If the actuals have not been calculated, apply the 
            self.test_data = test_data
            self.test_model(test_data)
        elif len(self.test_actuals) == 0:
            raise Exception("No test data is available, please provide test data!")

        self.test_model(test_data)
        # Total sum of squares (denominator)
        y = self.test_data.iloc[:, 0].to_numpy()
        
        rmse = np.sqrt(np.sum((y - self.test_actuals)**2)/len(y))

        self.rmse = rmse
        return self.rmse
    
    def increment_training_counter(self):
        self.generation += 1

    def get_training_counter(self):
        return self.generation
        
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
                
                axes[id].plot(cur_param_list, deg_list, label=cur_set.name)

            axes[id].legend()
                
        plt.show()


if __name__ == '__main__':
    # Test features
    test_dict = {0 : (0, 6), 1 : (0, 6)}
    test_names = ["A", "B"]
    test_tsk_model = TSKModel()
    #test_tsk_model.randomize_model(4, 4*4, test_dict, test_names, max_expressions=2)

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

    new_rulebase = TSKRuleBase(test_names, {"A" : "test1", "B" : "test2"})
    possible_params = [[1,1],[0,0], [1,1], [0,0]]
    possible_r = [0,2,-1,9]
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


