import matplotlib.pyplot as plt
# matplotlib inline

f = plt.figure()
plt.errorbar(
    [5,10,15,20,25,30],  # X
    [0.969302,0.970216,0.971766,0.970672,0.970214,0.96784], # Y
    yerr=[0.000880153,0.000876772,0.001225553,0.001005803,0.000813929,0.001842186],     # Y-errors
    label="Text GCN",
    fmt="ro--", # format line like for plot()
    linewidth=2	# width of plot line
    )


plt.legend() #Show legend
plt.show()

f.savefig("results/window_R8.pdf", bbox_inches='tight')

f = plt.figure()
plt.errorbar(
    [5,10,15,20,25,30],  # X
    [0.76078,0.765792,0.768718,0.767395,0.766748,0.767084], # Y
    yerr=[0.001863752,0.002426813,0.000597512,0.001961939,0.000910396,0.003008393],     # Y-errors
    label="Text GCN",
    fmt="ro--", # format line like for plot()
    linewidth=2	# width of plot line
    )


plt.legend() #Show legend
plt.show()

f.savefig("results/window_MR.pdf", bbox_inches='tight')