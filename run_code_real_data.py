from functions_for_application import *

for idx in range(4):
    with open(f'application/fold_{idx}.pkl', 'rb') as f:
        data = pickle.load(f)
    table = {}
    run_code(data, table, idx)
    
