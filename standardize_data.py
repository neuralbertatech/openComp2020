# Standardize "Back":
# Back has a big dip in channel 2, To find the dip, find the minimum point.
# Throw away any data that has a minimum within x of the avg
# Throw away any data where the point falls outside of the middle 75% (x0.125x | 0.75 | x0.125x)
# Center all channels on this data point
# Shorten window len to 0.5s to ensure no missing data

# So far this approach is not going to work, the data is too variable.
# If we throw away suing the rules above, we will be rejecting about 70%
# of our data, and we will only be able to record the right thing 30% of the time.
