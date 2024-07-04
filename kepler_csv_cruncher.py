import math
import re

kepler_raw = open(r"cumulative_2024.06.26_12.15.36.csv", "r")
kepler_str = kepler_raw.read()
kepler_raw.close()

kepler_str = kepler_str.split('\n')
kepler_str = kepler_str[21:-1]  # removing the header stuff and the last blank row
kepler_pure = []
for S in kepler_str:
    temp = S.split(',')
    kepler_pure.append(temp[1:])  # I don't know that words can express how annoyed I am with NOAA nor how pleased I am with NASA, for the brevity of this code vs the other one.

# KepID, Ground Truth, orbital period, transit duration, transit depth, body/star rad ratio, body radius, orbit sMa, inclination, body temperature, body insolation flux, star temperature, star radius, star mass

kepler_filtered = []

for P in kepler_pure:
    temparray = []
    temparray.append(P[0])
    temparray.append(P[1])
    temparray.append(P[2:])
    kepler_filtered.append(temparray)

# at this point, kepler_filtered consists of a list of lists of lists
# A: list of all observation-packages B
# B: KepID, Ground Truth (confirmed vs candidate vs false positive), list of observational data C
# C: orbital period, transit duration, transit depth, body/star rad ratio, body radius, orbit sMa, inclination, body temperature, body insolation flux, star temperature, star radius, star mass