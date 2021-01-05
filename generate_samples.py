import numpy as np
import pandapower as pp
from define_network import define_network
import random as rand
from sklearn import mixture
import pypsa

def generate_samples(net_case,n_of_network_samples,net, percent_of_measurements, estimation_method, gmm, net_type="pp"):
    
    if net_type == "pp":

        injection_values, network_state_samples, measurement_vector = np.zeros((n_of_network_samples,net.bus.shape[0])),\
        np.zeros((n_of_network_samples,net.bus.shape[0])), np.zeros((n_of_network_samples,int(net.bus.shape[0]*percent_of_measurements)))

        measurement_indices = rand.sample(range(net.bus.shape[0]),int(net.bus.shape[0]*percent_of_measurements))

        print(range(net.bus.shape[0]))

        print(measurement_indices)

        for i in range(n_of_network_samples):

            net = define_network(net_case)

            net.ext_grid.va_degree = 10.0

            net.line.loc[net.line.r_ohm_per_km == 0,'r_ohm_per_km'] = 0.1

            injection_values_per_iter = 2e-3*np.squeeze(gmm.sample(n_samples=net.bus.shape[0])[0])

            injection_values[i,:] = injection_values_per_iter

            k = 0

            # Need to figure out to to get value parameter for line current measurements, since the data only show max voltage.

            # For now, working with voltage measurements

            measurements = np.zeros((net.bus.shape[0],))

            for j in range(net.bus.shape[0]):

                ref_bus_voltage = net.bus.vn_kv[j]

                measurement_value = np.abs(ref_bus_voltage + rand.normalvariate(0,0.1*ref_bus_voltage))

                #current_elements = np.random.choice(net.line.index, size=net.bus.shape[0])

                pp.create_measurement(net=net, meas_type='v', element_type='bus', value=ref_bus_voltage, std_dev=0.1*ref_bus_voltage, element=j, side=None, check_existing=True, index=None, name=None)

                #pp.create_measurement(net=net, meas_type='i', element_type='line', value=ref_bus_voltage, std_dev=0.1*ref_bus_voltage, element=line1, side=None, check_existing=True, index=None, name=None)

                #if j != net.bus.shape[0]-1:

                #pp.create_measurement(net=net, meas_type='q', element_type='bus', value=injection_values_per_iter[j], std_dev=0.05*ref_bus_voltage, element=j, side=None, check_existing=True, index=None, name=None)

                pp.create_measurement(net=net, meas_type='p', element_type='bus', value=injection_values_per_iter[j], std_dev=0.1*ref_bus_voltage, element=j, side=None, check_existing=True, index=None, name=None)

                pp.create_sgen(net=net, bus=j, p_mw=injection_values_per_iter[j])

            pp.diagnostic(net, report_style='detailed', warnings_only=True)

            pp.runpp(net)

            if estimation_method =='standard':

                success = pp.estimation.estimate(net, init="flat", tolerance=1e-3, maximum_iterations=100, calculate_voltage_angles=False)
                V, delta = net.res_bus_est.vm_pu, net.res_bus_est.va_degree

            else:

                success = pp.estimation.remove_bad_data(net, init="flat",tolerance=1e-3,maximum_iterations=100, calculate_voltage_angles=False)
                V_rn_max, delta_rn_max = net.res_bus_est.vm_pu, net.res_bus_est.va_degree


            #     success_chi2 = pp.estimation.chi2_analysis(net, init="flat")

            #print("Z: ", net.res_bus.vm_pu.values)

            network_state_samples[i,:] = net.res_bus.vm_pu.values

            measurement_vector[i,:] = measurements[measurement_indices] #net.res_bus_est.vm_pu.values[measurement_indices]

        return injection_values, network_state_samples, measurement_vector
    
    elif net_type == "pypsa_3":
        
        measurement_indices = rand.sample(range(net.buses.shape[0]),int(net.buses.shape[0]*percent_of_measurements))
        
#         print("meas_indexes:", measurement_indices)
        
        injection_values, network_state_samples, measurement_vector = np.zeros((n_of_network_samples,net.buses.shape[0])),\
        np.zeros((n_of_network_samples,net.buses.shape[0])), np.zeros((n_of_network_samples,int(net.buses.shape[0]*percent_of_measurements)))

        measurement_indices = rand.sample(range(net.buses.shape[0]),int(net.buses.shape[0]*percent_of_measurements))

#         print(range(net.buses.shape[0]))

#         print(measurement_indices)

        for i in range(n_of_network_samples):
            
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
                
            injection_values_per_iter = np.squeeze(gmm.sample(n_samples=net.buses.shape[0])[0])

            injection_values[i,:] = injection_values_per_iter

            #add a generator at bus 0
            net.add("Generator","My gen",
                        bus="My bus 0",
                        p_set=10,
                        control="PQ")


            for i in range(n_buses):
                #add a load at bus 1
                net.add("Load","My load {}".format(i),
                            bus="My bus {}".format(i),
                            p_set=injection_values_per_iter[i])

            net.loads.q_set = 10.

            net.pf()
            
            network_state_samples[i,:] = net.buses_t.v_mag_pu.values
            
            measurements = np.abs(net.buses_t.v_mag_pu.values + np.random.normal(0, 0.1*5, net.buses.shape[0]))[0]
            
            measurement_vector[i,:] = measurements[measurement_indices]

        return injection_values, network_state_samples, measurement_vector
    
    elif net_type == "opf-storage-hvdc":
        
        measurement_indices = rand.sample(range(net.buses.shape[0]),int(net.buses.shape[0]*percent_of_measurements))
        
#         print("meas_indexes:", measurement_indices)
        
        injection_values, network_state_samples, measurement_vector = np.zeros((n_of_network_samples,net.buses.shape[0])),\
        np.zeros((n_of_network_samples,net.buses.shape[0])), np.zeros((n_of_network_samples,int(net.buses.shape[0]*percent_of_measurements)))

        measurement_indices = rand.sample(range(net.buses.shape[0]),int(net.buses.shape[0]*percent_of_measurements))

#         print(range(net.buses.shape[0]))

#         print(measurement_indices)

        for i in range(n_of_network_samples):
            
            net = pypsa.Network(csv_folder_name='opf-storage-hvdc/opf-storage-data')

            #add three buses
            n_buses = net.buses.shape[0]
                
            injection_values_per_iter = np.squeeze(gmm.sample(n_samples=net.buses.shape[0])[0])

            injection_values[i,:] = injection_values_per_iter

            #add a generator at bus 0
            net.add("Generator","My gen",
                        bus="0",
                        p_set=10,
                        control="PQ")


            for i in range(n_buses):
                #add a load at bus 1
                net.add("Load","My load {}".format(i),
                            bus="{}".format(i),
                            p_set=injection_values_per_iter[i])

            net.loads.q_set = 10.

            net.pf()
            
            try:
            
                network_state_samples[i,:] = net.buses_t.v_mag_pu.values

                measurements = np.abs(net.buses_t.v_mag_pu.values + np.random.normal(0, 0.1*5, net.buses.shape[0]))
            
            except:
                
                network_state_samples[i,:] = net.buses_t.v_mag_pu.values[0]

                measurements = np.abs(net.buses_t.v_mag_pu.values[0] + np.random.normal(0, 0.1*5, net.buses.shape[0]))           
            
            measurement_vector[i,:] = measurements[measurement_indices]

        return injection_values, network_state_samples, measurement_vector
