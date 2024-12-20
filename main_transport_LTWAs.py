"""
running csptlib
# csptlib is a library of tools for evaluating the performance of arterial coordination,
    mainly based on vehicle trajectory data.
# The trajectory data can be used to directly output coordination ratings and comprehensive status information
    of vehicles travelling on trunk lines.
# The coordination effect of the arterial line can be studied visually by means of distance-time diagrams.
# Pudu diagrams can be output based on trajectory data to clarify the signal status of vehicles when they arrive
    at intersections.

Unfortunately my time is limited, later I will upload detailed descriptions of the use of csptlib.
copyright: CheneyZhao.wust.edu
           cheneyzhao@126.com
           # Used for impacts under example LTWAs.
"""
from matplotlib import pyplot as plt

from rateperformance import RateCoordinationPerformance, output_stop_delay, output_POG, output_stop_percent
from rateperformance import output_average_speed
from datastore import load_variavle, AterialDataCollection
from trajectoriesplot import SignalPlanPlot, PurduePlot

"""
The trajectory data files were simply too large to upload to GitHub, 
and we uploaded the ratings data at master  for use in the example.
"""
def exp(i, ari_signal_plan_ring, cycle, offset, phase):
    print(f'=======================================Scenario {i}=====================================================')
    # Data preparation
    # Intersection location information
    inter_location_inbound = [154.42, 1271.24, 1998.41, 2580.95]
    inter_location_outbound = [202.49, 1347.52, 2083.13, 2683.34]
    # ==========================================plot trajectory==================================================
    # tra = SignalPlanPlot()
    # tra.period = 3600
    # tra.gw_speed = 50
    # tra.phase_inbound, tra.phase_outbound = phase[0], phase[1]
    # tra.controllers_num, tra.cycle, tra.offset = 4, cycle, offset
    # tra.lanelength_inbound, tra.lanelength_outbound = [3018.30], [3018.30]
    # tra.inter_location_inbound, tra.inter_location_outbound = inter_location_inbound, inter_location_outbound
    # tra.ari_signal_plan_ring1 = ari_signal_plan_ring[0]
    # tra.ari_signal_plan_ring2 = ari_signal_plan_ring[1]
    # Load = AterialDataCollection()
    # tra.ari_signal_plan_ring1_color = Load.set_signal_color(phase_inbound, controllers_num_=4)
    # tra.ari_signal_plan_ring2_color = Load.set_signal_color(phase_outbound, controllers_num_=4)

    # tra.ari_signal_plan_ring1_hatch = Load.set_left_signal_hatch(phase_inbound, controllers_num_=4)
    # tra.ari_signal_plan_ring2_hatch = Load.set_left_signal_hatch(phase_outbound, controllers_num_=4)
    # tra.plot_signal_plan(band_text=(True, 1800))
    # #
    # inbound_trajectorydata = load_variavle(f'D:\\data\\TrajectoryData\\inbound_trajectorydata{i}.pkl')
    # outbound_trajectorydata = load_variavle(f'D:\\data\\TrajectoryData\\outbound_trajectorydata{i}.pkl')
    # tra.plot_trajectories(inbound_trajectorydata, outbound_trajectorydata)
    # tra.set_title(title=f'Scenario{i}', fontsize=25)
    # plt.savefig(f'D:\\data\\Scenario{i}.png', format='png')
    # # plt.show()
    # plt.close('all')
    # =============================================rate=====================================================
    # Read the trajectory data
    inbound_rate_data = load_variavle(f'D:\\data\\RateData\\inbound_rate_data{i}.pkl')
    outbound_rate_data = load_variavle(f'D:\\data\\RateData\\outbound_rate_data{i}.pkl')
    # ======================output average speed=========================================================
    output_average_speed(inbound_rate_data, f'inbound_avespeed{i}', 'D:\\data')
    output_average_speed(outbound_rate_data, f'outbound_avespeed{i}', 'D:\\data')
    # ======================output delay=========================================================
    output_stop_delay(inbound_rate_data, 4, f'inbound{i}_delay', 'D:\\data')
    output_stop_delay(outbound_rate_data, 4, f'outbound{i}_delay', 'D:\\data')
    # ========================evaluation==========================================
    ratedata_ = (inbound_rate_data, outbound_rate_data)
    inter_location_ = (inter_location_inbound, inter_location_outbound)
    lane_arterial_ = ([3, 6, 8, 6], [3, 7, 7, 6])
    lane_side_ = [6, 10, 10, 3]
    inter_traffic_volume_ = ([1311, 866, 1430, 1806], [1480, 1788, 1012, 1790])
    ari_traffic_volume_ = (1311, 1790)
    per = RateCoordinationPerformance(ratedata_, inter_location_, inter_traffic_volume_, ari_traffic_volume_,
                                      lane_arterial_, lane_side_)
    per.cycle = cycle
    per.ari_signal_plan_ring1 = ari_signal_plan_ring[0]  # Timing plan
    per.ari_signal_plan_ring2 = ari_signal_plan_ring[1]  # Timing plan
    per.output_performance_grade()
    # ========================output POG and stop ratio=========================================================
    # pu = PurduePlot(cycle=cycle, offset=offset, lanelength_outbound=3018.30)
    # inter_location_inbound = [154.42, 1271.24, 1998.41, 2580.95]
    # inter_location_outbound = [202.49, 1347.52, 2083.13, 2683.34]
    # pu.inter_location_inbound = inter_location_inbound
    # pu.inter_location_outbound = inter_location_outbound
    # pu.inbound_tradata = inbound_trajectorydata
    # pu.outbound_tradata = outbound_trajectorydata
    # pu.ari_signal_plan_ring1 = ari_signal_plan_ring[0]
    # pu.ari_signal_plan_ring2 = ari_signal_plan_ring[1]
    # pu.phase_inbound = phase[0]
    # pu.phase_outbound = phase[1]
    # # plot purdue diagram if need
    # # pu.plot_purdue()
    # # save vehicle arrival data
    # (purdue_data1, purdue_data2) = load_variavle(f'D:\\data\\PurdueData\\purdue_data{i}.pkl')
    # (purdue_BOG_EOG1, purdue_BOG_EOG2) = load_variavle(f'D:\\data\\PurdueData\\purdue_BOG_EOG{i}.pkl')
    # pog1 = output_POG(purdue_data1, purdue_BOG_EOG1)
    # pog2 = output_POG(purdue_data2, purdue_BOG_EOG2)
    # print('-------------------------------------------------------------------')
    # print(f'The average POG for Scenario {i} is:', (pog1 + pog2) / 2)
    # print('-------------------------------------------------------------------')
    # stop_r1 = output_stop_percent(inter_location_inbound, inbound_trajectorydata, 0)
    # stop_r2 = output_stop_percent(inter_location_outbound, outbound_trajectorydata, 0)
    # print('-------------------------------------------------------------------')
    # print(f'The average stop ratio for scenario {i} is:', (stop_r1 + stop_r2) / 2)
    # print('-------------------------------------------------------------------')


for S in [1, 2, 3, 4, 5]:  # Scenario
    phase_inbound, phase_outbound, offset_ = [], [], []
    ari_signal_plan_ring1, ari_signal_plan_ring2 = {}, {}
    # The reason for the discrepancy between the phase difference here and in the paper is that
    # VISSIM has to adjust the phase difference during the left turn phase front,
    # i.e. superimpose a left turn phase time.
    if S == 1:
        offset_ = [0, 96, 15, 61]
        phase_inbound = ['lead', 'lag', 'lag', 'lead']
        phase_outbound = ['lag', 'lag', 'lag', 'lag']
        ari_signal_plan_ring1 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [29.0, 43.0, 43.0, 52.0],
                                 'yellow_allred': [2.0, 2.0, 2.0, 2.0],
                                 'green2': [39.0, 37.0, 38.0, 50.0], 'red': [75.0, 64.0, 63.0, 42.0]}
        ari_signal_plan_ring2 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [43.0, 39.0, 25.0, 39.0],
                                 'yellow_allred': [2.0, 2.0, 2.0, 2.0],
                                 'green2': [25.0, 41.0, 56.0, 63.0], 'red': [75.0, 64.0, 63.0, 42.0]}
    elif S == 2:
        offset_ = [0, 56, 15, 57]
        phase_inbound = ['lead', 'lead', 'lag', 'lead']
        phase_outbound = ['lag', 'lead', 'lag', 'lag']
        ari_signal_plan_ring1 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [29.0, 41.0, 43.0, 52.0],
                                 'yellow_allred': [2.0, 2.0, 2.0, 2.0],
                                 'green2': [39.0, 39.0, 38.0, 50.0], 'red': [75.0, 64.0, 63.0, 42.0]}
        ari_signal_plan_ring2 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [43.0, 45.0, 25.0, 39.0],
                                 'yellow_allred': [2.0, 2.0, 2.0, 2.0],
                                 'green2': [25.0, 35.0, 56.0, 63.0], 'red': [75.0, 64.0, 63.0, 42.0]}
    elif S == 3:
        offset_ = [0, 96, 15, 61]
        phase_inbound = ['lead', 'lag', 'lag', 'lead']
        phase_outbound = ['lag', 'lag', 'lag', 'lag']
        ari_signal_plan_ring1 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [29.0, 43.0, 43.0, 52.0],
                                 'yellow_allred': [2.0, 2.0, 2.0, 2.0],
                                 'green2': [39.0, 37.0, 38.0, 50.0], 'red': [75.0, 64.0, 63.0, 42.0]}
        ari_signal_plan_ring2 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [43.0, 39.0, 25.0, 39.0],
                                 'yellow_allred': [2.0, 2.0, 2.0, 2.0],
                                 'green2': [25.0, 41.0, 56.0, 63.0], 'red': [75.0, 64.0, 63.0, 42.0]}
    elif S == 4:
        offset_ = [0, 56, 15, 57]
        phase_inbound = ['lead', 'lead', 'lag', 'lead']
        phase_outbound = ['lag', 'lead', 'lag', 'lag']
        ari_signal_plan_ring1 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [29.0, 41.0, 43.0, 52.0],
                                 'yellow_allred': [2.0, 2.0, 2.0, 2.0],
                                 'green2': [39.0, 39.0, 38.0, 50.0], 'red': [75.0, 64.0, 63.0, 42.0]}
        ari_signal_plan_ring2 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [43.0, 45.0, 25.0, 39.0],
                                 'yellow_allred': [2.0, 2.0, 2.0, 2.0],
                                 'green2': [25.0, 35.0, 56.0, 63.0], 'red': [75.0, 64.0, 63.0, 42.0]}
    elif S == 5:
        offset_ = [0, 62, 5, 60]
        phase_inbound = ['lead', 'lead', 'lead', 'lead']
        phase_outbound = ['lag', 'lead', 'lag', 'lag']
        ari_signal_plan_ring1 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [29.0, 41.0, 41.0, 51.0],
                                 'yellow_allred': [2.0, 2.0, 3.0, 3.0],
                                 'green2': [39.0, 39.0, 39.0, 50.0], 'red': [75.0, 64.0, 63.0, 42.0]}
        ari_signal_plan_ring2 = {'barrier1': [3.0, 2.0, 2.0, 2.0], 'green1': [43.0, 45.0, 25.0, 39.0],
                                 'yellow_allred': [2.0, 2.0, 2.0, 2.0],
                                 'green2': [25.0, 35.0, 56.0, 63.0], 'red': [75.0, 64.0, 63.0, 42.0]}
    cycle_ = [148, 148, 148, 148]
    phase_ = (phase_inbound, phase_outbound)
    ari_signal_plan_ring_ = (ari_signal_plan_ring1, ari_signal_plan_ring2)
    exp(S, ari_signal_plan_ring_, cycle_, offset_, phase_)
