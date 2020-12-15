
import pandapower.networks as pn

# Define Network

def define_network(net_case):
    
    if net_case == 'case_57':

        net = pn.case57()

    elif net_case == 'case_14s':

        net= pn.case14s()

    else:

        net = pn.case57()
        
    return net
