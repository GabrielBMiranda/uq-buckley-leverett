import numpy as np

def randon_variable_pdf(data, number_of_bins):
    # generates the histogram
    freq, bin_edges = np.histogram(data, bins=number_of_bins)
    # remove last value from bin border
    bins = bin_edges[:-1]
    # bin width
    binwidth = (data.max() - data.min()) / (number_of_bins - 1)
    # normalize data
    freq = freq / (data.size * binwidth)
    # calculates histogram area, should be fairly close to 1.0 as it was normalized
    area = binwidth * sum(freq)

    return bins, freq, area