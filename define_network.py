
import pandapower.networks as pn
import pypsa

# Define Network

def define_network(net_case):
    
    if net_case == 'case_57':

        net = pn.case57()

    elif net_case == 'case_14s':

        net= pn.case14s()

    elif net_case == 'case_3':

        net = pypsa.Network()

        #add three buses
        n_buses = 3

        for i in range(n_buses):
            net.add("Bus","My bus {}".format(i),
                        v_nom=5.)

        for i in range(n_buses):
            net.add("Line","My line {}".format(i),
                        bus0="My bus {}".format(i),
                        bus1="My bus {}".format((i+1)%n_buses),
                        x=0.1,
                        r=0.01)
            
    elif net_case == 'opf-storage-hvdc':

        net = pypsa.Network(csv_folder_name='opf-storage-hvdc/opf-storage-data')

    else:
        
        print("Invalid network type!")
        
        raise ValueError
        
    return net
