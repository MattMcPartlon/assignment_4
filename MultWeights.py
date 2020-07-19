import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod
class MultiplicativeWeights(ABC):

    def __init__(self, beta = 0.5):
        self.beta = beta
        self.experts = []
        self.weights = None

    def add_expert(self, expert):
        self.experts.append(expert)

    def reset_weights(self):
        self.weights = np.ones(len(self.experts))

    def get_predictions(self, data):
        return np.array([e.predict(data) for e in self.experts])

    def get_weighted_majority(self, preds):
        """
        implement this method
        :param preds: expert's predictions
        :return: a value in {\pm 1}
        """
        #TODO
        pass

    @abstractmethod
    def update_weights(self, preds, outcome):
        pass

    def scale_weights(self):
        max_wt = np.max(self.weights)
        self.weights-=max_wt
        self.weights+=1
        min_wt = np.min(self.weights)
        #FIXED
        if min_wt<2**(-20):
            self.weights[self.weights<2**(-20)]=2**(-20)

    def get_outcome(self, data :  Dict[str,Any], t : int):
        assert 0 < t < len(data['open'])
        if data['close'][t-1]>data['open'][t]:
            return -1
        return 1

    def get_reward(self, data, t, decision):
        outcome = self.get_outcome(data, t)
        abs_rew = abs(data['close'][t-1]-data['open'][t])
        return outcome*decision*abs_rew

    def get_data_up_to_t(self, data : Dict[str,Any], t : int):
        return {key : data[key][:t] for key in data}

    def get_data_from_t(self, data : Dict[str,Any], t : int):
        return {key : data[key][t:] for key in data}

    def get_data_in_range(self, data, start, end):
        temp = self.get_data_up_to_t(data,end)
        temp = self.get_data_from_t(temp,start)
        return temp

    def get_mistakes(self, data, start_day, decisions):
        T = len(data['open']) - start_day
        ground_truth = [self.get_outcome(data, start_day + t) for t in range(T)]
        ground_truth, decisions = np.array(ground_truth),np.array(decisions)
        return len(ground_truth[ground_truth!=decisions])

    def get_rewards(self, data, start_day, decisions):
        T = len(data['open']) - start_day
        return sum([self.get_reward(data, start_day+t, decisions[t]) for t in range(T)])


    def alg(self, data, start_day = 20):
        self.reset_weights()
        total_mistakes = 0
        expert_mistakes = np.zeros(len(self.experts))
        T = len(data['open'])-start_day
        decisions = []

        for t in range(T):
            #get the data for days 1...start_day + t - 1
            data_to_t = self.get_data_up_to_t(data, start_day+t)
            #have experts make predictions based on this data,
            #and get the weighted opinion
            preds = self.get_predictions(data_to_t)
            decision = self.get_weighted_majority(preds)
            #reveal the ground truth
            outcome = self.get_outcome(data, start_day+t)
            print(start_day+t)
            print(len(data['open']))
            decisions.append(decision)
            expert_mistakes[preds!=outcome]+=1
            if outcome!=decision:
                total_mistakes+=1
            if t%5 == 0: #avoid underflow
                self.scale_weights()
        reward = self.get_rewards(data, start_day, decisions)
        mistakes = self.get_mistakes(data, start_day, decisions)
        return decisions, expert_mistakes, mistakes, reward




