# -*- coding: utf-8 -*-
"""
Title: Rb Decay

The purpose of this code is to read in activity data of Rb from a file and fits it
to an expected curve, finding the decay constants for Rb and Sr (which decays into Rb),
it plots the data and the expected curve. This code also plots an contour plot
showing the chi squared values for different Rb and Sr decay constants.

Last updated: 9/12/2022
@author: UID: 10819085
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fmin
from scipy.constants import Avogadro


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Times New Roman'

# Constants---------------------------------------------------------------------
ORIGINAL_NUMBER_OF_SR_ATOMS = 10**(-6)*Avogadro
RB_START = 0.0005
SR_START = 0.005
DURATION_HOURS = 1
# Functions---------------------------------------------------------------------


def read_file(file_name):
    """
    Reads in data from a file removing any values that are not a number or
    are zero, and checking there are 3 columns of data

    Parameters
    ----------
    file_name : string

    Returns
    -------
    data : 2D Array

    """

    try:
        data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
        delete_1 = np.where(np.isnan(data).any(axis=1))
        delete_2 = np.where(data == 0)
        delete = np.append(delete_1[0], delete_2[0])
        data = np.delete(data, delete, axis=0)
    except OSError:
        print('File ', file_name, ' not found')

    no_columns = (len(data[0]))
    if no_columns != 3:
        print('FILE: ', file_name,
              ' is of the wrong format, file must have 3 columns')
        print('Either correct current file or choose another and run code again')
        sys.exit()

    return data


def hours_to_seconds(hours_time):
    """
    Function converts time in hours to seconds

    Parameters
    ----------
    hours_time : integer or float

    Returns
    -------
    60*60*hours_time: integer or float

    """
    return 60*60*hours_time


def activity_function(decay_constants, time):
    """
    Function calculates the activity of a sample with specific decay constanst
    at a specific time

    Parameters
    ----------
    decay_constants : tuple (containting floats)
    time : integer or float

    Returns
    -------
    activity : float

    """
    time = hours_to_seconds(time)
    rb_decay_constant = decay_constants[0]
    sr_decay_constant = decay_constants[1]
    activity = rb_decay_constant*(ORIGINAL_NUMBER_OF_SR_ATOMS)*(sr_decay_constant/(
        rb_decay_constant-sr_decay_constant))*(
            np.exp(-sr_decay_constant*time)-np.exp(-rb_decay_constant*time))
    activity = activity*10**-12
    return activity


def chi_squared(decay_constants, time, y_value, y_uncertainty):
    """
    Calculates the chi squared of a fit compared to our data

    Parameters
    ----------
    decay_constants :  tuple (containting floats)
    time : array or float
    y : array or float
    y_uncertainty : array or float

    Returns
    -------
    chi : float

    """
    y_fit = activity_function(decay_constants, time)
    chi = np.sum(((y_fit-y_value)/y_uncertainty)**2)

    return chi


def half_life(decay_constant):
    """
    Finds the half life (in minutes) of an element based on its decay constant

    Parameters
    ----------
    decay_constant : float

    Returns
    -------
    half_life_calculated : float

    """
    half_life_calculated = np.log(2)/decay_constant
    half_life_calculated = half_life_calculated/60
    return half_life_calculated


def plot(data, plot_name, y_expected, time, title):
    """
    Plots the ideal curve and the data

    Parameters
    ----------
    data : array
    plot_name : string
    y_expected : array
    time : array
    title : string
    Returns
    -------
    None.

    """

    fig = plt.figure()

    axis = fig.add_subplot(111)

    axis.errorbar(data[:, 0], data[:, 1], data[:, 2],
                  fmt='.', color='pink', label='Recorded data')
    axis.plot(time, y_expected, color='m',
              label='Function of expected activity')

    axis.set_title(title)
    axis.set_xlabel('Time (hours)')
    axis.set_ylabel('Activity (TBq)')
    axis.legend()
    plt.savefig(plot_name, facecolor='white', edgecolor='k', format='png')

    plt.show()


def filter_data(data, coefficients):
    """
    Filters out anomalies from the data

    Parameters
    ----------
    data : 2D array
    coefficients : tuple(containg floats)

    Returns
    -------
    new_data:2D array

    """

    new_data = np.zeros((0, 3))
    for x_val in range(0, len(data)):
        activity_expected = activity_function((coefficients), data[x_val, 0])

        if np.abs(data[x_val, 1]-activity_expected) > 3*data[x_val, 2] or data[x_val, 2] == 0:
            new_data = new_data+0

        else:
            new_data = np.vstack((new_data, data[x_val]))

    return new_data


def reduced_chi(degrees_of_freedom, chi_squared_in):
    """
    Finds the rediced chi squared of the fit

    Parameters
    ----------
    degrees_of_freedom : integer
    chi_squared : Tfloat

    Returns
    -------
    reduced: float

    """
    reduced = chi_squared_in/degrees_of_freedom
    return reduced


def data_handling(data_set, plot_name, plot_title, time):
    """
    Finds the decay constant values that produce a minimum chi squared and
    calls the plot function

    Parameters
    ----------
    data_set : 2D array
    plot_name : string
    plot_title : string
    time : array

    Returns
    -------
    fit_result : tuple

    """
    fit_result = fmin(chi_squared, (RB_START, SR_START), full_output=True,
                      args=(data_set[:, 0], data_set[:, 1], data_set[:, 2]))
    y_expected = activity_function(fit_result[0], time)

    plot(data_set, plot_name,
         y_expected, time, plot_title)
    return fit_result


def contour_plot_function(coefficients, data):
    """
    This functions plots a contour plot of the chi squared value
    against the coefficient values

    Parameters
    ----------
    coefficients : tuple (containing floats)
    data : array
    chi_official : float

    Returns
    -------
    meshx : array
    meshy : array
    meshz : array

    """
    x_values = np.linspace(
        coefficients[0]-(coefficients[0]*0.03), coefficients[0]+(coefficients[0]*0.03), 500)
    y_values = np.linspace(
        coefficients[1]-(coefficients[1]*0.03), coefficients[1]+(coefficients[1]*0.03), 500)

    meshx, meshy = np.meshgrid(x_values, y_values)

    meshz = np.zeros((0, len(meshx)))
    for y_no in range(0, len(meshx)):
        line = np.array([])
        for x_no in range(0, len(meshx)):

            chi = chi_squared((meshx[y_no, x_no], meshy[y_no, x_no]),
                              data[:, 0], data[:, 1], data[:, 2])
            line = np.append(line, chi)
        meshz = (np.vstack((meshz, line)))

    fig = plt.figure()
    axis = fig.add_subplot(111)

    contour_plot = axis.contour(meshx, meshy, meshz, 15, colors='k')
    contourf_plot = axis.contourf(meshx, meshy, meshz, 100, cmap='plasma')
    fig.colorbar(contourf_plot)

    axis.scatter(coefficients[0], coefficients[1], color='k',
                 marker='*', label='minimum chi squared value')
    axis.legend()
    axis.set_title('Contour plot of chi squared values')
    axis.set_xlabel('Rb constant')
    axis.set_ylabel('Sr constant')
    axis.clabel(contour_plot, colors='k')

    plt.savefig('Contour_plot.png', facecolor='white',
                edgecolor='k', format='png')
    plt.show()
    return meshx, meshy, meshz


def uncertainty_constants(value_location, mesh):
    """
    THis function finds the uncertainty on the decay constants

    Parameters
    ----------
    value_location : tuple
    mesh : 2D array

    Returns
    -------
    uncertainty_value : float

    """
    list_values = np.array([])

    for i in range(0, len(value_location[0])):
        list_values = np.append(
            list_values, mesh[value_location[0][i], value_location[1][i]])

    maximum = np.amax(list_values)
    minimum = np.amin(list_values)
    uncertainty_value = (maximum-minimum)/2

    return uncertainty_value


def uncertainty_time(decay_uncertainty, decay_constant, half_life_value):
    """
    Calculates the uncertainty in the half life

    Parameters
    ----------
    decay_uncertainty : float
    decay_constant : float
    half_life_value : float

    Returns
    -------
    half_life_uncertainty : float

    """
    half_life_uncertainty = (decay_uncertainty/decay_constant)*half_life_value
    return half_life_uncertainty


def uncertainty_activity(constant1, uncertainty1, constant2, uncertainty2, time, activity):
    """
    Calculates the uncertainty in the activity

    Parameters
    ----------
    constant1 : float
    uncertainty1 : float
    constant2 : float
    uncertainty2 : float
    time : float
    activity : float

    Returns
    -------
    f_value_uncertainty :float

    """
    time = hours_to_seconds(time)
    a_value = constant1-constant2
    a_value_uncertainty = np.sqrt(
        np.square(uncertainty1)+np.square(uncertainty2))
    b_value = constant2/a_value
    b_value_uncertainty = b_value * np.sqrt(np.square(a_value_uncertainty/(
        a_value)) + np.square(uncertainty2/constant2))
    d_value = np.exp(-constant2*time)
    d_value_uncertainty = d_value*(uncertainty2/constant2)
    e_value = np.exp(-constant1*time)
    e_value_uncertainty = e_value*(uncertainty1/constant1)
    c_value = d_value-e_value
    c_value_uncertainty = np.sqrt(
        (e_value_uncertainty**2)+(d_value_uncertainty**2))
    f_value_uncertainty = activity * np.sqrt(np.square(
        c_value_uncertainty/c_value) + np.square(b_value_uncertainty/b_value))

    return f_value_uncertainty


def activity_inquiry(decay_constant1, decay_constant2, decay_uncertainty1, decay_uncertainty2):
    """
    Calculates the activity at a time specified by the user

    Parameters
    ----------
    decay_constant1 : float
    decay_constant2 : float
    decay_uncertainty1 : float
    decay_uncertainty2 : float


    Returns
    -------
    answer : string

    """
    try:
        value = float(input(
            'What is the time (in minutes) you would like to know the activity of? '))

        activity_calculated = activity_function(
            (decay_constant1, decay_constant2), value/60)
        uncertainty = uncertainty_activity(
            decay_constant1, decay_uncertainty1, decay_constant2,
            decay_uncertainty2, value, activity_calculated)
        if np.isnan(uncertainty):
            print('[Value too large,enter another smaller value]')
            answer = activity_inquiry(decay_constant1, decay_constant2,
                                      decay_uncertainty1, decay_uncertainty2)
        elif uncertainty < 0.001:
            print(
                'The activity at {} minutes is {:.3g} TBq, the uncertainty is negligable'.format(
                    value, activity_calculated))
            answer = input(
                'Do you wish to know the activity at another time(yes/no)? ')
        else:
            print('The activity at {} minutes is {:.3g} +/- {:.2g} TBq'.format(
                value, activity_calculated, uncertainty))

            answer = input(
                'Do you wish to know the activity at another time(yes/no)? ')
    except ValueError:
        print('Incorrect value type')
        answer = activity_inquiry(decay_constant1, decay_constant2,
                                  decay_uncertainty1, decay_uncertainty2)

    return answer


def strontium_activity(decay_const_sr, time_sr):
    """
    Calculates strontium activity at a specified time

    Parameters
    ----------
    decay_const_sr : float
    time_sr : float

    Returns
    -------
    sr_activity : float

    """
    time_sr = hours_to_seconds(time_sr)
    sr_activity = decay_const_sr*ORIGINAL_NUMBER_OF_SR_ATOMS * \
        (np.exp(-decay_const_sr*time_sr))

    return sr_activity


def sr_activity_plot(sr_const):
    """
    Plots the Strontium actvity against time

    Parameters
    ----------
    sr_const : flaot

    Returns
    -------
    None.

    """
    x_axis = np.linspace(0, DURATION_HOURS, 1000)
    y_axis = strontium_activity(sr_const, x_axis)

    fig = plt.figure()

    axis = fig.add_subplot(111)

    axis.plot(x_axis, y_axis, color='m')

    axis.set_title('Strontium activity over time')
    axis.set_xlabel('Time (hours)')
    axis.set_ylabel('Activity (Bq)')
    plt.savefig('Strontium activity plot.png', facecolor='white',
                edgecolor='k', format='png')
    plt.show()


def main():
    """
    Function contains the main code of the programme

    Returns
    -------
    None.

    """

    nuclear_data_set1 = read_file('Nuclear_data_1.csv')
    nuclear_data_set2 = read_file('Nuclear_data_2.csv')
    nuclear_data_set3 = np.vstack((nuclear_data_set1, nuclear_data_set2))
    print(len(nuclear_data_set3))
    time_array = np.linspace(0, DURATION_HOURS, 10000)

    fit_results_unfiltered = data_handling(
        nuclear_data_set3, 'Nuclear_data3_plot_unfiltered.png',
        'Unfiltered data plot: Rubidium activity',
        time_array)

    nuclear_data_set3_filtered = filter_data(
        nuclear_data_set3, fit_results_unfiltered[0])

    fit_results_filtered = data_handling(
        nuclear_data_set3_filtered, 'Nuclear_data3_plot_filtered.png',
        'Filtered data plot: Rubidium activity',
        time_array)

    rb_decay_constant, sr_decay_constant = fit_results_filtered[0]

    rb_half_life = half_life(rb_decay_constant)
    sr_half_life = half_life(sr_decay_constant)

    chi_squared_fit = (fit_results_filtered[1])

    reduce_chi_squared = reduced_chi(
        (len(nuclear_data_set3_filtered)-2), chi_squared_fit)

    activity_at_90mins = activity_function(fit_results_filtered[0], 1.5)

    x_mesh, y_mesh, z_mesh = contour_plot_function(
        fit_results_filtered[0], nuclear_data_set3_filtered)

    z_mesh = np.round_(z_mesh, 3)

    locations = np.where(z_mesh == np.round_(chi_squared_fit+1, 3))

    rb_constant_uncertainty = uncertainty_constants(locations, x_mesh)
    sr_constant_uncertainty = uncertainty_constants(locations, y_mesh)

    rb_half_life_uncertainty = uncertainty_time(
        rb_constant_uncertainty, rb_decay_constant, rb_half_life)

    sr_half_life_uncertainty = uncertainty_time(
        sr_constant_uncertainty, sr_decay_constant, sr_half_life)

    activity_90mins_uncertainty = uncertainty_activity(
        rb_decay_constant, rb_constant_uncertainty, sr_decay_constant,
        sr_constant_uncertainty, 1.5, activity_at_90mins)

    if reduce_chi_squared < 0.5 or reduce_chi_squared > 2:
        print('It is unlikely this data matches our function at all')
        print('It is probable another function describes this data,'
              ' or this data has significant outliers and should be measured again')

    print('The decay constant of Rubidium is {:.2e} +/- {:.1e} s^-1s'.format(
        rb_decay_constant, rb_constant_uncertainty))
    print('The half-life of Rubiduium is {:.3g} +/- {:.3g} minutes'.format(
        rb_half_life, rb_half_life_uncertainty))

    print('The decay constant of Strontium is {:.2e} +/- {:.1e} s^-1 '.format(
        sr_decay_constant, sr_constant_uncertainty))
    print('The half-life of Strontium is {:.3g} +/- {:.3g} minutes'.format(
        sr_half_life, sr_half_life_uncertainty))

    print('The chi squared of our fit is {:.2f}'.format(
        chi_squared_fit))

    print('The reduced chi squared of our fit is {:.2f}'.format(
        reduce_chi_squared))

    print('At 90 minutes the activity is {:.3g} +/- {:.2g} TBq'.format(
        activity_at_90mins, activity_90mins_uncertainty))

    user_answer = input(
        'Do you wish to know the activity at another time(yes/no)? ')

    while user_answer.upper() == 'YES':
        user_answer = activity_inquiry(rb_decay_constant, sr_decay_constant,
                                       rb_constant_uncertainty, sr_constant_uncertainty,)

    user_answer2 = input(
        'Do you wish to see the activity of strontium over time (yes/no) ? ')
    if user_answer2.upper() == 'YES':
        sr_activity_plot(sr_decay_constant)


# Main Code---------------------------------------------------------------------

main()
