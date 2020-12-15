import numpy as np
import pandapower as pp
from define_network import define_network
import random as rand
from sklearn import mixture

def generate_samples(net_case,n_of_network_samples,net, percent_of_measurements, estimation_method, gmm):

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

			success = pp.estimation.estimate(net, init="flat")
			V, delta = net.res_bus_est.vm_pu, net.res_bus_est.va_degree

		else:
		    
			success = pp.estimation.remove_bad_data(net, init="flat")
			V_rn_max, delta_rn_max = net.res_bus_est.vm_pu, net.res_bus_est.va_degree


		#     success_chi2 = pp.estimation.chi2_analysis(net, init="flat")

		#print("Z: ", net.res_bus.vm_pu.values)

		network_state_samples[i,:] = net.res_bus.vm_pu.values

		measurement_vector[i,:] = measurements[measurement_indices] #net.res_bus_est.vm_pu.values[measurement_indices]

	return injection_values, network_state_samples, measurement_vector
