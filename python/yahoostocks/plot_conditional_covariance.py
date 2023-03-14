"""
=======================================
Visualizing the stock market structure
=======================================

This example employs several unsupervised learning techniques to extract
the stock market structure from variations in historical quotes.

The quantity that we use is the daily variation in quote price: quotes
that are linked tend to fluctuate in relation to each other during a day.
"""

# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause

# %%
# Retrieve the data from Internet
# -------------------------------
#
# The data is from 2003 - 2008. This is reasonably calm: (not too long ago so
# that we get high-tech firms, and before the 2008 crash). This kind of
# historical data can be obtained from APIs like the quandl.com and
# alphavantage.co .

import sys
import time

import numpy as np
import pandas as pd

from python.yahoostocks.yahoostock import YahooStock

symbol_dict_BAK = {
    "SIVB": "Silicon Valley Bank",
    "XOM": "Exxon",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "VLO": "Valero Energy",
    "MSFT": "Microsoft",
    "IBM": "IBM",
    "TWX": "Time Warner",
    "CMCSA": "Comcast",
    "CVC": "Cablevision",
    "YHOO": "Yahoo",
    "DELL": "Dell",
    "HPQ": "HP",
    "AMZN": "Amazon",
    "TM": "Toyota",
    "CAJ": "Canon",
    "SNE": "Sony",
    "F": "Ford",
    "HMC": "Honda",
    "NAV": "Navistar",
    "NOC": "Northrop Grumman",
    "BA": "Boeing",
    "KO": "Coca Cola",
    "MMM": "3M",
    "MCD": "McDonald's",
    "PEP": "Pepsi",
    "K": "Kellogg",
    "UN": "Unilever",
    "MAR": "Marriott",
    "PG": "Procter Gamble",
    "CL": "Colgate-Palmolive",
    "GE": "General Electrics",
    "WFC": "Wells Fargo",
    "JPM": "JPMorgan Chase",
    "AIG": "AIG",
    "AXP": "American express",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "AAPL": "Apple",
    "SAP": "SAP",
    "CSCO": "Cisco",
    "TXN": "Texas Instruments",
    "XRX": "Xerox",
    "WMT": "Wal-Mart",
    "HD": "Home Depot",
    "GSK": "GlaxoSmithKline",
    "PFE": "Pfizer",
    "SNY": "Sanofi-Aventis",
    "NVS": "Novartis",
    "KMB": "Kimberly-Clark",
    "R": "Ryder",
    "GD": "General Dynamics",
    "RTN": "Raytheon",
    "CVS": "CVS",
    "CAT": "Caterpillar",
    "DD": "DuPont de Nemours",
}

symbol_dict_LONG = {
    "SIVB": "Silicon Valley Bank",
    "XOM": "Exxon",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "VLO": "Valero Energy",
    "MSFT": "Microsoft",
    "IBM": "IBM",
    "CMCSA": "Comcast",
    "DELL": "Dell",
    "HPQ": "HP",
    "AMZN": "Amazon",
    "TM": "Toyota",
    "CAJ": "Canon",
    "F": "Ford",
    "HMC": "Honda",
    "NOC": "Northrop Grumman",
    "BA": "Boeing",
    "KO": "Coca Cola",
    "MMM": "3M",
    "MCD": "McDonald's",
    "PEP": "Pepsi",
    "K": "Kellogg",
    "MAR": "Marriott",
    "PG": "Procter Gamble",
    "CL": "Colgate-Palmolive",
    "GE": "General Electrics",
    "WFC": "Wells Fargo",
    "JPM": "JPMorgan Chase",
    "AIG": "AIG",
    "AXP": "American express",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "AAPL": "Apple",
    "SAP": "SAP",
    "CSCO": "Cisco",
    "TXN": "Texas Instruments",
    "XRX": "Xerox",
    "WMT": "Wal-Mart",
    "HD": "Home Depot",
    "GSK": "GlaxoSmithKline",
    "PFE": "Pfizer",
    "SNY": "Sanofi-Aventis",
    "NVS": "Novartis",
    "KMB": "Kimberly-Clark",
    "R": "Ryder",
    "GD": "General Dynamics",
    "CVS": "CVS",
    "CAT": "Caterpillar",
    "DD": "DuPont de Nemours",
}


# for symbol in symbols:
#     print("Fetching quote history for %r" % symbol, file=sys.stderr)
#     url = (
#         "https://raw.githubusercontent.com/scikit-learn/examples-data/"
#         "master/financial-data/{}.csv"
#     )
#     quotes.append(pd.read_csv(url.format(symbol)))

symbol_dict_SHORT = {
    # Cluster 1
    "SIVB": "Silicon Valley Bank",
    "CAJ": "Canon",
    "AMZN": "Amazon",
    "CVX": "Chevron",
    "CAT": "Caterpillar",
    "SCHW": "Charles Schwab",
    "BA": "Boeing",
    "WMT": "Wal-Mart",
    "AXP": "American Express",
    # Cluster 2
    "PEP": "Pepsi",
    "BAC": "Bank of America",
    "AIG": "AIG",
    "TM": "Toyota",
    "CMCSA": "Comcast",
    "MMM": "3M",
    "COP": "ConocoPhillips",
    "CL": "Colgate-Palmolive",
    "NOC": "Northrop Grumman",
    "WFC": "Wells Fargo",
    # Cluster 3
    "AAPL": "Apple",
    "DELL": "Dell",
    "KEY": "KeyCorp",
    "R": "Ryder",
    "JPM": "JPMorgan Chase",
    # Cluster 4
    "KMB": "Kimberly-Clark",
    "CSCO": "Cisco",
    "PFE": "Pfizer",

    ### Uncluster
    # "MCD": "McDonald's",
    # "HPQ": "HP",
    # "F": "Ford",
    # "SNY": "Sanofi-Aventis",
    # "CVS": "CVS",
    # "DD": "DuPont de Nemours",
    # "XRX": "Xerox",
    # "XOM": "Exxon",
    # "KO": "Coca Cola",
    # "PG": "Procter Gamble",
    # "IBM": "IBM",
    # "MAR": "Marriott",
    # "GS": "Goldman Sachs",
    # "HMC": "Honda",
    # "K": "Kellogg",
    # "GSK": "GlaxoSmithKline",
    # "MSFT": "Microsoft",
    # "GD": "General Dynamics",
    # "FRC": "First Republic Bank",
    # "TXN": "Texas Instruments",
    # "SAP": "SAP",
    # "VLO": "Valero Energy",
    # "HD": "Home Depot",
    # "NVS": "Novartis",
}

### CHANGE THE VARIABLE BELOW
active_data = symbol_dict_SHORT
### CHANGE NOTHING BELOW
NUM_NEIGHBORS = len(active_data)-2

for row in range(50, 10, -1):
    quotes = []

    # NUM_NEIGHBORS = len(active_data)-1

    #
    symbols, names = np.array(sorted(active_data.items())).T
    for ticker in active_data:
        print("starting ticker: " + ticker)
        yahoostock = YahooStock(ticker)
        these_quotes = yahoostock.price_frame.fillna(.000001)
        quotes.append(these_quotes.iloc[-row:])

    close_prices = np.vstack([q["close"] for q in quotes])
    open_prices = np.vstack([q["open"] for q in quotes])
    volume = np.vstack([q["volume"] for q in quotes])

    # The daily variations of the quotes are what carry the most information
    variation = (close_prices - open_prices)*volume

    # %%
    # .. _stock_market:
    #
    # Learning a graph structure
    # --------------------------
    #
    # We use sparse inverse covariance estimation to find which quotes are
    # correlated conditionally on the others. Specifically, sparse inverse
    # covariance gives us a graph, that is a list of connection. For each
    # symbol, the symbols that it is connected too are those useful to explain
    # its fluctuations.

    from sklearn import covariance

    alphas = np.logspace(-1.5, 1, num=10)
    edge_model = covariance.GraphicalLassoCV(alphas=alphas)

    # standardize the time series: using correlations rather than covariance
    # former is more efficient for structure recovery

    # Seems to be a divive by 0 error in X /= X.std(axis=0)
    X = variation.copy().T
    pd_T = pd.DataFrame(X)
    X /= X.std(axis=0)
    X = np.nan_to_num(X, copy=False, nan=0.00001)
    edge_model.fit(X)

    # %%
    # Clustering using affinity propagation
    # -------------------------------------
    #
    # We use clustering to group together quotes that behave similarly. Here,
    # amongst the :ref:`various clustering techniques <clustering>` available
    # in the scikit-learn, we use :ref:`affinity_propagation` as it does
    # not enforce equal-size clusters, and it can choose automatically the
    # number of clusters from the data.
    #
    # Note that this gives us a different indication than the graph, as the
    # graph reflects conditional relations between variables, while the
    # clustering reflects marginal properties: variables clustered together can
    # be considered as having a similar impact at the level of the full stock
    # market.

    from sklearn import cluster

    _, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=9, damping=.8, max_iter=10000)
    n_labels = labels.max()

    for i in range(n_labels + 1):
        print(f"Cluster {i + 1}: {', '.join(names[labels == i])}")

    # %%
    # Embedding in 2D space
    # ---------------------
    #
    # For visualization purposes, we need to lay out the different symbols on a
    # 2D canvas. For this we use :ref:`manifold` techniques to retrieve 2D
    # embedding.
    # We use a dense eigen_solver to achieve reproducibility (arpack is initiated
    # with the random vectors that we don't control). In addition, we use a large
    # number of neighbors to capture the large-scale structure.

    # Finding a low-dimension embedding for visualization: find the best position of
    # the nodes (the stocks) on a 2D plane

    from sklearn import manifold

    node_position_model = manifold.LocallyLinearEmbedding(
        n_components=2, eigen_solver="dense", n_neighbors=NUM_NEIGHBORS
    )

    embedding = node_position_model.fit_transform(X.T).T

    # %%
    # Visualization
    # -------------
    #
    # The output of the 3 models are combined in a 2D graph where nodes
    # represents the stocks and edges the:
    #
    # - cluster labels are used to define the color of the nodes
    # - the sparse covariance model is used to display the strength of the edges
    # - the 2D embedding is used to position the nodes in the plan
    #
    # This example has a fair amount of visualization-related code, as
    # visualization is crucial here to display the graph. One of the challenge
    # is to position the labels minimizing overlap. For this we use an
    # heuristic based on the direction of the nearest neighbor along each
    # axis.

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    plt.figure(1, facecolor="w", figsize=(10, 8))
    plt.clf()
    current_figure = plt.gcf()

    ax = plt.axes([0.0, 0.0, 1.0, 1.0])
    plt.axis("off")

    # Plot the graph of partial correlations
    partial_correlations = edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0.02

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(
        embedding[0], embedding[1], s=100 * d**2, c=labels, cmap=plt.cm.nipy_spectral
    )

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [
        [embedding[:, start], embedding[:, stop]] for start, stop in zip(start_idx, end_idx)
    ]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(
        segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, 0.7 * values.max())
    )
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(zip(names, labels, embedding.T)):

        dx = x - embedding[0]
        dx[index] = 1
        dy = y - embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = "left"
            x = x + 0.002
        else:
            horizontalalignment = "right"
            x = x - 0.002
        if this_dy > 0:
            verticalalignment = "bottom"
            y = y + 0.002
        else:
            verticalalignment = "top"
            y = y - 0.002
        plt.text(
            x,
            y,
            name,
            size=10,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            bbox=dict(
                facecolor="w",
                edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                alpha=0.6,
            ),
        )

    plt.xlim(
        embedding[0].min() - 0.15 * embedding[0].ptp(),
        embedding[0].max() + 0.10 * embedding[0].ptp(),
    )
    plt.ylim(
        embedding[1].min() - 0.03 * embedding[1].ptp(),
        embedding[1].max() + 0.03 * embedding[1].ptp(),
    )

    plt.show()
    plt.draw()
    file_name = "./" + str(row)+ "_days_ago.png"
    plt.savefig(file_name)
    time.sleep(2)

