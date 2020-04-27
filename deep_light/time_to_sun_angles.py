

#
#
#  Copyright (c) 2020.  Yue Liu
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  If you find this code useful please cite:
#  Predicting Annual Equirectangular Panoramic Luminance Maps Using Deep Neural Networks,
#  Yue Liu, Alex Colburn and and Mehlika Inanici.  16th IBPSA International Conference and Exhibition, Building Simulation 2019.
#
#
#



import math

# Calculates sun angles (azimuth and altitude) from given information (latitude, time and date)
# Refer from IES Handbook, 7.1.5 Solar Position AND Radiance sun.cal file

#expect julian date 0- 365
def calc_ET(JULIANDATE):
    return 0.170 * math.sin(4*math.pi / 373 *(JULIANDATE - 80) ) - 0.129 * math.sin(2* math.pi / 355 *(JULIANDATE - 8))

#expect julian date 0-365
def calc_solarDeclination(JulianDate):
    return 0.4093 * math.sin(2*math.pi / 368 * (JulianDate - 81) )

# expect standard time in decimal hours (0-24), ET(equation of time) between -14 and 16,
# SM standard meridian for the time zone, and site longtitude
def calc_solarTime(standardTime, ET, SM, longtitude):
    return standardTime + ET + 12 * (SM - longtitude) / math.pi


def calc_azimuth(solarDeclination, latitude, solarTime):
    return -math.atan2(math.cos(solarDeclination) * math.sin(math.pi*solarTime/12)
                     ,-math.cos(latitude)*math.sin(solarDeclination) - math.sin(latitude)*math.cos(solarDeclination)*math.cos(math.pi*solarTime/12))

def calc_altitude(latitude, solarDeclination, solarTime):
    return math.asin(math.sin(latitude) * math.sin(solarDeclination)
                     - math.cos(latitude) * math.cos(solarDeclination) * math.cos(math.pi * solarTime /12))

# read date and month, time in decimal, latitude and longtitude in degree, return azimuth and altitude in degree
def timeToAltitudeAzimuth(date, month, time, latitude, longtitude, standard_medirian):
    DATE = date
    MONTH = month
    STANDARDTIME = time
    LATITUDE = latitude * math.pi / 180
    LONGTITUDE = longtitude * math.pi / 180

    MONTH_LIST = [0,31,59,90,120,151,181,212,243,273,304,334]
    JULIANDATE = MONTH_LIST[MONTH-1] + DATE
    SM = standard_medirian  * math.pi / 180 #PST TIME ZONE is 120
    ET = calc_ET(JULIANDATE)
    SOLARTIME = calc_solarTime(STANDARDTIME, ET, SM, LONGTITUDE)
    SOLARDECLINATION = calc_solarDeclination(JULIANDATE)
    ALTITUDE = calc_altitude(LATITUDE, SOLARDECLINATION, SOLARTIME) * 180 / math.pi
    AZIMUTH = calc_azimuth(SOLARDECLINATION, LATITUDE, SOLARTIME) * 180/ math.pi
    return (ALTITUDE, AZIMUTH)


# location related
LATITUDE = 47
LONGTITUDE = 122
SM = 120

# time wises
MONTH = 12
DATE = 31
STANDARDTIME = 9.5

results = timeToAltitudeAzimuth(DATE, MONTH, STANDARDTIME, LATITUDE, LONGTITUDE, SM)
print(results)
