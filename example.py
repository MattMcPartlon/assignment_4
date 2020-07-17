
from assignment_4.Expert import MeanReversion, YesterdaysNews, ExpectationExpert
from assignment_4.MutiplicativeWeights1 import MultiplicativeWeights1 as MW1
import numpy as np
"""
Example code for adding experts and running the multiplicative weights algorithm
"""

"""
Load the data
"""

#setup data
path_to_data = '???/assignment_4/data/stock_data.npy'
stock_data = np.load(path_to_data,allow_pickle=True).item()

for key1 in stock_data:
    print(key1)
    for key2 in stock_data[key1]:
        print(key2)

starting_day = 50
max_window = min(starting_day-1, 20)
experts = list()
experts.append(YesterdaysNews())

"""
create experts considering data from various previous time frames
"""
for window in range(1,max_window):
    e1 = MeanReversion(window, sign = 1)
    e2 = ExpectationExpert(window, sign =1)
    experts += [e1, e2]
    #e1 = MeanReversion(window, sign = -1)
    #e2 = ExpectationExpert(window, sign =-1)
    #experts+=[e1, e2]

#add experts to multiplicative weights algorithm

mw = MW1()
for e in experts:
    mw.add_expert(e)

temp = stock_data['MSFT']
data = mw.get_data_in_range(temp,0,300)
out = mw.alg(data,start_day=starting_day)
decisions, expert_mistakes, mistakes, reward = out
print('num experts :',len(experts))
print('reward :',reward)
print('mistakes :',mistakes)
print('percentage of mistakes',100*mistakes/len(decisions))
print('min of expert mistakes',np.min(expert_mistakes))


